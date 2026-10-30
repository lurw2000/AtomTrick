"""Equivariant linear projection for O3Tensors.

By Schur's lemma, a linear map that commutes with all O(3) group
representations must act as an independent learnable matrix on the
*multiplicity* axis of each (l, p) block, while leaving the (2l+1)
magnetic components untouched.  This module implements exactly that.

The forward pass is a **single dense matrix-vector multiply** — all
per-sector weights are expanded into a block-diagonal weight matrix
at construction time (index pattern) and assembled on-the-fly via
one ``index_put`` + one ``F.linear``.  No Python loops at inference.

Both :class:`O3Linear` and :class:`O2Linear` support two modes:

* ``internal=True`` (default) — weights and bias are stored as
  ``nn.Parameter`` and learned normally.
* ``internal=False`` — no parameters are stored.  The caller must
  pass a flat weight vector (and optionally bias) to ``forward()``.
  Use ``weight_numel`` / ``bias_numel`` to query the required sizes.
  This is useful for hypernetworks or dynamic weight generation.

Usage::

    # Standard (internal weights)
    linear = O3Linear("4x0e + 4x1o", "8x0e + 8x1o")
    y = linear(x)

    # External weights
    linear = O3Linear("4x0e + 4x1o", "8x0e + 8x1o", internal=False)
    w = some_hypernet(...)                   # shape (..., linear.weight_numel)
    y = linear(x, weight=w)                  # weight provided externally
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .irreps import Irrep, Irreps
from .tensor import O3Tensor
from .utils import register_index_buffer


@torch.jit.script
def _o3_linear_per_sector_core(
    raw: Tensor,
    w: Tensor,
    in_start: List[int],
    mi_list: List[int],
    out_start: List[int],
    mo_list: List[int],
    dim_list: List[int],
    w_off: List[int],
    out_dim: int,
    bias: Optional[Tensor],
    bias_start: int,
    bias_stop: int,
) -> Tensor:
    """Scripted per-(l,p) sector matmul.

    For each shared sector ``i``:
      ``x_block  = raw[..., in_start[i] : in_start[i] + mi_list[i]*dim_list[i]]``
                   .reshape(*batch, mi_list[i], dim_list[i])
      ``W_lp    = w[w_off[i] : w_off[i] + mo_list[i]*mi_list[i]].view(mo_list[i], mi_list[i])``
      ``y_block = W_lp @ x_block``   (broadcasts batch)
      ``out[..., out_start[i] : out_start[i] + mo_list[i]*dim_list[i]] = y_block.reshape(..., mo_list[i]*dim_list[i])``

    No dense (out_dim, in_dim) matrix is ever materialised.  The scripted
    loop amortises Python dispatch over ``len(in_start)`` sectors — which
    is typically ≤ lmax+1.
    """
    batch_shape = raw.shape[:-1]
    out = raw.new_zeros(batch_shape + (out_dim,))
    n_sectors = len(in_start)
    for i in range(n_sectors):
        mi = mi_list[i]
        mo = mo_list[i]
        dim = dim_list[i]
        off_i = in_start[i]
        off_o = out_start[i]
        off_w = w_off[i]
        W_lp = w.narrow(0, off_w, mo * mi).view(mo, mi)
        x_slice = raw.narrow(-1, off_i, mi * dim)
        x_block = x_slice.reshape(batch_shape + (mi, dim))
        y_block = torch.matmul(W_lp, x_block)  # (*batch, mo, dim)
        y_flat = y_block.reshape(batch_shape + (mo * dim,))
        out.narrow(-1, off_o, mo * dim).copy_(y_flat)
    if bias is not None and bias_stop > bias_start:
        bslice = out.narrow(-1, bias_start, bias_stop - bias_start)
        bslice.add_(bias)
    return out


@torch.jit.script
def _o3_linear_per_sector_core_blocks(
    raw: Tensor,
    ws: List[Tensor],                   # one (mo, mi) matrix per shared sector
    mo_list: List[int],
    mi_list: List[int],
    in_start: List[int],
    out_start: List[int],
    dim_list: List[int],
    out_dim: int,
    bias: Optional[Tensor],
    bias_start: int,
    bias_stop: int,
) -> Tensor:
    """Per-(l,p) sector matmul reading BLOCK parameters directly.

    Identical arithmetic to ``_o3_linear_per_sector_core`` but the weights
    arrive as a list of ``(mo, mi)`` matrices instead of one flat vector, so the
    ``narrow`` + ``view`` pair per sector disappears.  ``mo`` / ``mi`` are read
    from each block's own shape.
    """
    batch_shape = raw.shape[:-1]
    out = raw.new_zeros(batch_shape + (out_dim,))
    n_sectors = len(in_start)
    for i in range(n_sectors):
        W_lp = ws[i]
        mo = mo_list[i]  # Python ints, not W.size(): symbolic under make_fx (compile ~10x slower)
        mi = mi_list[i]
        dim = dim_list[i]
        x_slice = raw.narrow(-1, in_start[i], mi * dim)
        x_block = x_slice.reshape(batch_shape + (mi, dim))
        y_block = torch.matmul(W_lp, x_block)          # (*batch, mo, dim)
        y_flat = y_block.reshape(batch_shape + (mo * dim,))
        out.narrow(-1, out_start[i], mo * dim).copy_(y_flat)
    if bias is not None and bias_stop > bias_start:
        bslice = out.narrow(-1, bias_start, bias_stop - bias_start)
        bslice.add_(bias)
    return out


class _FlatWeightCompat:
    """Mixin: expose the legacy flat ``.weight`` on a block-parameter module.

    The equivariant linears used to store one flat 1-D ``nn.Parameter``.  They
    now store a ``ParameterList`` of dense blocks (so a matrix-aware optimiser
    sees real matrices, and the forward drops a narrow+view per block).  Two compatibility surfaces are kept:

    * ``.weight`` -- a PROPERTY returning the flat concatenation.  Read-only,
      for tests, analysis and the ``weight_numel`` contract.  It is NOT used on
      the forward hot path.
    * a ``load_state_dict`` pre-hook that splits a legacy flat ``weight`` entry
      into ``weight_blocks.{i}``, so every checkpoint written before this change
      still loads unchanged.
    """

    @property
    def weight(self) -> Optional[Tensor]:
        wb = getattr(self, "weight_blocks", None)
        if wb is None or len(wb) == 0:
            return None
        return torch.cat([b.reshape(-1) for b in wb])

    def _register_flat_weight_compat(self, block_shapes: List[Tuple[int, ...]]):
        self._block_shapes = [tuple(int(d) for d in s) for s in block_shapes]
        self._register_load_state_dict_pre_hook(
            _split_flat_weight_hook, with_module=True)


def _split_flat_weight_hook(module, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    """Rewrite a legacy flat ``<prefix>weight`` into ``<prefix>weight_blocks.i``."""
    key = prefix + "weight"
    if key not in state_dict:
        return
    flat = state_dict.pop(key)
    shapes = getattr(module, "_block_shapes", None)
    if not shapes:
        return
    expect = sum(int(torch.tensor(s).prod()) for s in shapes)
    if flat.numel() != expect:
        error_msgs.append(
            "%sweight: legacy flat weight has %d elements but the block layout "
            "needs %d -- refusing to guess a split."
            % (prefix, flat.numel(), expect))
        return
    off = 0
    for i, shape in enumerate(shapes):
        n = 1
        for d in shape:
            n *= d
        state_dict[prefix + "weight_blocks." + str(i)] = \
            flat[off:off + n].reshape(shape)
        off += n


class O3Linear(_FlatWeightCompat, nn.Module):
    """Equivariant linear projection between two O3Tensor irrep layouts.

    For each (l, p) sector present in *both* ``irreps_in`` and ``irreps_out``,
    a weight matrix of shape ``(mult_out, mult_in)`` is learned.  The (2l+1)
    magnetic components are shared (broadcast), so the map commutes with
    Wigner D-matrix rotations by construction.

    Irrep sectors present in ``irreps_in`` but absent from ``irreps_out`` are
    projected away.  Sectors in ``irreps_out`` but absent from ``irreps_in``
    are zero-filled.

    Parameters
    ----------
    irreps_in : str | list | Irreps
        Input irrep specification.
    irreps_out : str | list | Irreps
        Output irrep specification.
    bias : bool
        If ``True``, add a learnable bias to the scalar (l=0, p=+1) sector
        only — bias on higher-l components would break equivariance.
        Default: ``True``.
    internal : bool
        If ``True`` (default), weights and bias are stored as learnable
        ``nn.Parameter``.  If ``False``, no parameters are created —
        call ``forward(x, weight=..., bias=...)`` with external tensors.
        Query ``weight_numel`` and ``bias_numel`` for required sizes.
    """

    def __init__(
        self,
        irreps_in: Union[str, list, Irreps],
        irreps_out: Union[str, list, Irreps],
        *,
        bias: bool = True,
        internal: bool = True,
    ):
        super().__init__()
        if not isinstance(irreps_in, Irreps):
            irreps_in = Irreps(irreps_in)
        if not isinstance(irreps_out, Irreps):
            irreps_out = Irreps(irreps_out)

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.internal = internal

        # Simplify: merge duplicate (l,p) entries so each sector is contiguous.
        in_simple = irreps_in.simplified()
        out_simple = irreps_out.simplified()

        # Build sector multiplicities
        in_mults: Dict[Tuple[int, int], int] = {}
        for mult, irrep in in_simple:
            in_mults[(irrep.l, irrep.p)] = mult

        out_mults: Dict[Tuple[int, int], int] = {}
        for mult, irrep in out_simple:
            out_mults[(irrep.l, irrep.p)] = mult

        shared_keys = sorted(set(in_mults) & set(out_mults))

        # Build slices for simplified layouts
        in_slices: Dict[Tuple[int, int], slice] = {}
        offset = 0
        for mult, irrep in in_simple:
            width = mult * irrep.dim
            in_slices[(irrep.l, irrep.p)] = slice(offset, offset + width)
            offset += width

        out_slices: Dict[Tuple[int, int], slice] = {}
        offset = 0
        for mult, irrep in out_simple:
            width = mult * irrep.dim
            out_slices[(irrep.l, irrep.p)] = slice(offset, offset + width)
            offset += width

        # ---- Flat weight vector + index pattern for block-diagonal expansion ----

        weight_pieces: List[Tensor] = []
        weight_mats: List[Tensor] = []          # the same weights, unflattened
        row_indices: List[Tensor] = []
        col_indices: List[Tensor] = []

        sector_info: List[Tuple[int, int, int, int]] = []
        flat_offset = 0

        for l, p in shared_keys:
            mi, mo = in_mults[(l, p)], out_mults[(l, p)]
            dim = 2 * l + 1
            n_weights = mo * mi

            w = torch.empty(mo, mi)
            nn.init.normal_(w, std=1.0 / math.sqrt(mi))
            weight_pieces.append(w.reshape(-1))
            weight_mats.append(w)

            sector_info.append((flat_offset, mo, mi, dim))

            sl_in = in_slices[(l, p)]
            sl_out = out_slices[(l, p)]

            a_idx = torch.arange(mo).unsqueeze(1).expand(mo, mi).reshape(-1)
            b_idx = torch.arange(mi).unsqueeze(0).expand(mo, mi).reshape(-1)
            k_idx = torch.arange(dim)

            rows = (sl_out.start + a_idx.unsqueeze(1) * dim + k_idx.unsqueeze(0)).reshape(-1)
            cols = (sl_in.start + b_idx.unsqueeze(1) * dim + k_idx.unsqueeze(0)).reshape(-1)

            row_indices.append(rows)
            col_indices.append(cols)

            flat_offset += n_weights

        # Weight numel (always computed, even when internal=False)
        self._weight_numel = flat_offset

        # Weights are stored as one dense (mo, mi) block per (l, p) sector --
        # NOT as a flat vector -- so a matrix-aware optimiser sees real
        # matrices with no side-channel metadata, and the forward drops a
        # narrow+view per sector.  ``.weight`` still exposes the flat view via
        # _FlatWeightCompat, and legacy checkpoints are split on load.
        if internal:
            self.weight_blocks = nn.ParameterList(
                [nn.Parameter(w) for w in weight_mats])
        else:
            self.weight_blocks = None
        self._register_flat_weight_compat([tuple(w.shape) for w in weight_mats])

        # Register index tensors as (non-persistent) buffers.  _row_idx /
        # _col_idx / _repeats / _expand_idx describe the dense block-diagonal
        # expansion; the forward no longer uses them, but tests/analysis do.
        if row_indices:
            register_index_buffer(self, "_row_idx", torch.cat(row_indices))
            register_index_buffer(self, "_col_idx", torch.cat(col_indices))
        else:
            register_index_buffer(self, "_row_idx", torch.empty(0, dtype=torch.long))
            register_index_buffer(self, "_col_idx", torch.empty(0, dtype=torch.long))

        self._sector_info = sector_info
        self._in_dim = in_simple.dim
        self._out_dim = out_simple.dim

        # Per-(l,p) sector metadata for the block-sparse forward.
        # Stored as parallel ``List[int]`` fields (rather than a list of
        # tuples) so ``@torch.jit.script`` can consume them — TorchScript
        # only supports ``List[int]`` and not generic ``List[Tuple[...]]``.
        in_start_list: List[int] = []
        mi_list: List[int] = []
        out_start_list: List[int] = []
        mo_list: List[int] = []
        dim_list: List[int] = []
        w_off_list: List[int] = []
        for (l, p), info in zip(shared_keys, sector_info):
            mi = in_mults[(l, p)]
            mo = out_mults[(l, p)]
            dim = 2 * l + 1
            w_off = info[0]
            in_sl = in_slices[(l, p)]
            out_sl = out_slices[(l, p)]
            in_start_list.append(int(in_sl.start))
            mi_list.append(int(mi))
            out_start_list.append(int(out_sl.start))
            mo_list.append(int(mo))
            dim_list.append(int(dim))
            w_off_list.append(int(w_off))
        self._sec_in_start = in_start_list
        self._sec_mi = mi_list
        self._sec_out_start = out_start_list
        self._sec_mo = mo_list
        self._sec_dim = dim_list
        self._sec_w_off = w_off_list

        # Precompute expand-index: maps flat weight → dense block-diagonal entries.
        # Using a gather index avoids repeat_interleave at runtime.
        repeats: List[int] = []
        for l, p in shared_keys:
            dim = 2 * l + 1
            n_weights = in_mults[(l, p)] * out_mults[(l, p)]
            repeats.extend([dim] * n_weights)
        if repeats:
            register_index_buffer(self, "_repeats", torch.tensor(repeats, dtype=torch.long))
            _expand_idx = torch.arange(flat_offset).repeat_interleave(
                self._repeats)
            register_index_buffer(self, "_expand_idx", _expand_idx)
        else:
            register_index_buffer(self, "_repeats", torch.empty(0, dtype=torch.long))
            register_index_buffer(self, "_expand_idx", torch.empty(0, dtype=torch.long))


        # Bias
        self._bias_slice: Optional[slice] = None
        self._bias_numel = 0
        if bias and (0, 1) in out_mults:
            self._bias_numel = out_mults[(0, 1)]
            self._bias_slice = out_slices[(0, 1)]
            if internal:
                self.bias = nn.Parameter(torch.zeros(self._bias_numel))
            else:
                self.bias = None
        else:
            self.bias = None

        # Reorder indices
        need_in_reorder = (irreps_in != in_simple)
        need_out_reorder = (irreps_out != out_simple)

        if need_in_reorder:
            register_index_buffer(self, "_in_gather",
                                  self._build_gather_index(irreps_in, in_simple))
        else:
            self._in_gather = None

        if need_out_reorder:
            register_index_buffer(self, "_out_scatter",
                                  self._build_scatter_index(irreps_out, out_simple))
        else:
            self._out_scatter = None

    @property
    def weight_numel(self) -> int:
        """Number of weight scalars needed for the equivariant projection."""
        return self._weight_numel

    @property
    def bias_numel(self) -> int:
        """Number of bias scalars (0 if bias is disabled)."""
        return self._bias_numel

    @staticmethod
    def _build_gather_index(original: Irreps, simplified: Irreps) -> Tensor:
        indices: List[int] = []
        for _, irrep_s in simplified:
            lp = (irrep_s.l, irrep_s.p)
            for i, (_, irrep_o) in enumerate(original):
                if (irrep_o.l, irrep_o.p) == lp:
                    sl = original._slices[i]
                    indices.extend(range(sl.start, sl.stop))
        return torch.tensor(indices, dtype=torch.long)

    @staticmethod
    def _build_scatter_index(original: Irreps, simplified: Irreps) -> Tensor:
        scatter = torch.zeros(original.dim, dtype=torch.long)
        sim_offset = 0
        for _, irrep_s in simplified:
            lp = (irrep_s.l, irrep_s.p)
            for i, (_, irrep_o) in enumerate(original):
                if (irrep_o.l, irrep_o.p) == lp:
                    sl = original._slices[i]
                    width = sl.stop - sl.start
                    scatter[sl.start:sl.stop] = torch.arange(sim_offset, sim_offset + width)
                    sim_offset += width
        return scatter

    def forward(
        self,
        x: O3Tensor,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
    ) -> O3Tensor:
        """Apply equivariant linear projection.

        Parameters
        ----------
        x : O3Tensor
            Input tensor with ``x.irreps == self.irreps_in``.
        weight : Tensor, optional
            Flat weight vector of shape ``(..., weight_numel)``.  Required
            when ``internal=False``; if provided when ``internal=True`` it
            overrides the stored parameters.  Leading batch dims are
            broadcast with ``x``.
        bias : Tensor, optional
            Bias vector of shape ``(..., bias_numel)``.  Same semantics
            as *weight*.
        """
        if x.irreps != self.irreps_in:
            raise ValueError(
                f"Expected input irreps {self.irreps_in}, got {x.irreps}"
            )

        # Resolve weight.  An EXTERNALLY supplied weight is flat -- a
        # hypernetwork emits (..., weight_numel) -- so it keeps the flat core.
        # Internal weights are per-sector blocks and use the block core.
        if weight is None and self.weight_blocks is None:
            raise ValueError(
                "No weight provided and internal=False — pass weight to forward()"
            )

        # Resolve bias
        b = bias if bias is not None else self.bias

        raw = x.tensor

        # Rearrange to simplified order if needed
        if self._in_gather is not None:
            raw = raw[..., self._in_gather]

        bias_start = 0
        bias_stop = 0
        if self._bias_slice is not None:
            bias_start = self._bias_slice.start
            bias_stop = self._bias_slice.stop

        # Scripted per-(l,p) sector matmul — no dense (D, D) materialisation.
        if weight is not None:
            out = _o3_linear_per_sector_core(
                raw, weight,
                self._sec_in_start, self._sec_mi,
                self._sec_out_start, self._sec_mo,
                self._sec_dim, self._sec_w_off,
                self._out_dim,
                b, bias_start, bias_stop,
            )
        else:
            out = _o3_linear_per_sector_core_blocks(
                raw, list(self.weight_blocks), self._sec_mo, self._sec_mi,
                self._sec_in_start, self._sec_out_start,
                self._sec_dim, self._out_dim,
                b, bias_start, bias_stop,
            )

        # Rearrange back to original output order if needed
        if self._out_scatter is not None:
            out = out[..., self._out_scatter]

        return O3Tensor(out, self.irreps_out)

    def extra_repr(self) -> str:
        return (
            f"irreps_in={self.irreps_in}, "
            f"irreps_out={self.irreps_out}, "
            f"bias={'scalar' if self._bias_numel > 0 else 'none'}, "
            f"internal={self.internal}, "
            f"weight_numel={self._weight_numel}, "
            f"bias_numel={self._bias_numel}"
        )


@torch.jit.script
def _o2_linear_block_internal_core(
    raw: Tensor,
    w: Tensor,                          # 1-D flat weight
    block_in_idx: Tensor,               # (sum mi*ncomp,) gather indices
    block_out_idx: Tensor,              # (sum mo*ncomp,) scatter indices
    mo_list: List[int],
    mi_list: List[int],
    ncomp_list: List[int],
    w_off_list: List[int],
    in_off_list: List[int],
    out_off_list: List[int],
    out_dim: int,
) -> Tensor:
    """Per-|m|-block matmul with 1-D internal weights (no batch dim on W).

    Gather the (mi × n_comp) input channels of each block, apply the
    (mo × mi) block weight via a single ``torch.matmul`` that broadcasts
    over the batch and the n_comp axis, and scatter back into ``out``.
    Never materialises a dense (B, out_dim, in_dim) matrix.
    """
    batch_shape = raw.shape[:-1]
    out = raw.new_zeros(batch_shape + (out_dim,))
    n_blocks = len(mo_list)
    for i in range(n_blocks):
        mo = mo_list[i]
        mi = mi_list[i]
        nc = ncomp_list[i]
        wo = w_off_list[i]
        io = in_off_list[i]
        oo = out_off_list[i]

        in_idx = block_in_idx.narrow(0, io, mi * nc)
        out_idx = block_out_idx.narrow(0, oo, mo * nc)

        # (*batch, mi*nc) -> (*batch, mi, nc)
        x_blk = raw.index_select(-1, in_idx).reshape(batch_shape + (mi, nc))
        W_m = w.narrow(0, wo, mo * mi).view(mo, mi)
        # (mo, mi) @ (*batch, mi, nc) -> (*batch, mo, nc)
        y_blk = torch.matmul(W_m, x_blk)
        out.index_copy_(-1, out_idx, y_blk.reshape(batch_shape + (mo * nc,)))
    return out


@torch.jit.script
def _o2_linear_block_external_core(
    raw: Tensor,
    w: Tensor,                          # shape (*batch, weight_numel) — per-edge
    block_in_idx: Tensor,
    block_out_idx: Tensor,
    mo_list: List[int],
    mi_list: List[int],
    ncomp_list: List[int],
    w_off_list: List[int],
    in_off_list: List[int],
    out_off_list: List[int],
    out_dim: int,
) -> Tensor:
    """Per-|m|-block batched matmul with per-edge external weights.

    Identical structure to ``_o2_linear_block_internal_core`` but reshapes
    ``w`` as ``(*batch, mo, mi)`` per block and uses ``torch.matmul`` so
    the batched matmul runs natively — **no (B, D, D) allocation**.
    """
    batch_shape = raw.shape[:-1]
    out = raw.new_zeros(batch_shape + (out_dim,))
    n_blocks = len(mo_list)
    for i in range(n_blocks):
        mo = mo_list[i]
        mi = mi_list[i]
        nc = ncomp_list[i]
        wo = w_off_list[i]
        io = in_off_list[i]
        oo = out_off_list[i]

        in_idx = block_in_idx.narrow(0, io, mi * nc)
        out_idx = block_out_idx.narrow(0, oo, mo * nc)

        x_blk = raw.index_select(-1, in_idx).reshape(batch_shape + (mi, nc))
        W_m = w.narrow(-1, wo, mo * mi).reshape(batch_shape + (mo, mi))
        # (*batch, mo, mi) @ (*batch, mi, nc) -> (*batch, mo, nc)
        y_blk = torch.matmul(W_m, x_blk)
        out.index_copy_(-1, out_idx, y_blk.reshape(batch_shape + (mo * nc,)))
    return out


@torch.jit.script
def _o2_linear_block_internal_core_blocks(
    raw: Tensor,
    ws: List[Tensor],                   # one (mo, mi) matrix per |m| block
    mo_list: List[int],
    mi_list: List[int],
    block_in_idx: Tensor,
    block_out_idx: Tensor,
    ncomp_list: List[int],
    in_off_list: List[int],
    out_off_list: List[int],
    out_dim: int,
) -> Tensor:
    """Per-|m|-block matmul reading BLOCK parameters directly.

    Same arithmetic as ``_o2_linear_block_internal_core``, but the weights
    arrive as dense matrices instead of slices of a flat vector, so the
    ``narrow`` + ``view`` per block disappears.  ``mo`` / ``mi`` are passed as Python ints:
    read from the weight's shape they become symbolic under make_fx tracing, which made
    the compiled training step ~10x slower to compile.
    """
    batch_shape = raw.shape[:-1]
    out = raw.new_zeros(batch_shape + (out_dim,))
    n_blocks = len(ncomp_list)
    for i in range(n_blocks):
        W_m = ws[i]
        mo = mo_list[i]  # Python ints, not W.size(): symbolic under make_fx (compile ~10x slower)
        mi = mi_list[i]
        nc = ncomp_list[i]
        in_idx = block_in_idx.narrow(0, in_off_list[i], mi * nc)
        out_idx = block_out_idx.narrow(0, out_off_list[i], mo * nc)
        x_blk = raw.index_select(-1, in_idx).reshape(batch_shape + (mi, nc))
        y_blk = torch.matmul(W_m, x_blk)
        out.index_copy_(-1, out_idx, y_blk.reshape(batch_shape + (mo * nc,)))
    return out


def _check_natural_parity(irreps_in: Irreps, irreps_out: Irreps, cls_name: str) -> None:
    """Raise unless every irrep has natural parity ``p = (-1)**l``.

    O2Linear / SO2Linear act in the edge frame and mix, for each ``|m|``, the
    ``(l, m)`` components of *different* degrees ``l``.  A reflection through a
    plane that contains the edge axis multiplies the ``(l, m)`` component by a
    sign that depends on ``p * (-1)**l`` (and on ``m``), so mixing two degrees
    commutes with it -- i.e. the layer stays O(3)-equivariant once rotated back
    out of the edge frame -- only if ``p * (-1)**l`` is the same for every
    irrep.  Any layout that contains ``0e`` (AtomTrick's hidden irreps always
    do) must therefore be all natural parity (0e, 1o, 2e, 3o, ...), the
    convention used everywhere in AtomTrick; an all-pseudo-parity layout would
    also be consistent but is not used and is rejected as well.  Nothing
    checked this before, so a mixed layout (e.g. ``0e + 1e``) silently broke
    reflection equivariance.
    """
    bad = []
    for irreps in (irreps_in, irreps_out):
        for _mult, ir in irreps:
            if ir.p != (-1) ** ir.l and str(ir) not in bad:
                bad.append(str(ir))
    if bad:
        raise ValueError(
            f"{cls_name} requires natural-parity irreps, p = (-1)^l "
            f"(0e, 1o, 2e, 3o, ...); got {', '.join(bad)} in "
            f"irreps_in={irreps_in} / irreps_out={irreps_out}.  The edge-frame "
            f"map mixes the m-components of different degrees l that share the "
            f"same |m|; that mixing is consistent with reflections (O(3) "
            f"equivariance) only when p*(-1)^l is identical for every irrep.  "
            f"Use e.g. '64x0e+64x1o+64x2e+64x3o'.")


class O2Linear(_FlatWeightCompat, nn.Module):
    r"""O(2) equivariant linear projection between two O3Tensor layouts.

    Compared to :class:`O3Linear` (one weight per ``(l, p)`` sector), this
    layer has one weight per ``|m|`` sector.  Because only rotations around
    a single axis need to be respected, channels from **different** ``(l, p)``
    entries that share the same ``|m|`` are allowed to mix — giving the
    layer strictly more expressivity.

    **Representation theory.**  Under :math:`R_z(\alpha)`, the Wigner
    D-matrix is block-diagonal in ``|m|``:

    * ``m = 0`` — invariant (1-D block, one component per multiplicity).
    * ``|m| > 0`` — the ``(+m, -m)`` pair rotates by :math:`m\alpha`
      (2-D block per multiplicity).

    By Schur's lemma the equivariant linear map is block-diagonal in
    ``|m|``, with an independent ``(mult_out, mult_in)`` weight matrix
    per ``|m|`` value.  For ``|m| > 0`` the same weight is applied
    identically to the ``+m`` and ``-m`` components (Kronecker with
    :math:`I_2`), preserving the 2-D rotation structure.

    Mixing degrees within an ``|m|`` block is compatible with reflections
    only for natural-parity irreps (``p = (-1)**l``: 0e, 1o, 2e, 3o, ...);
    the constructor rejects any other layout (see ``_check_natural_parity``).

    The symmetry axis is specified at **forward time** via the ``axis``
    argument.  The layer rotates the coordinate frame so that the given
    axis aligns with +z, applies the O(2)_z equivariant linear map, and
    rotates back.  Batched per-sample axes ``(*, 3)`` are supported,
    making this suitable for per-edge projections in graph networks.
    When ``axis`` is ``None`` (default), the +z axis is assumed and no
    rotation is performed.

    Parameters
    ----------
    irreps_in, irreps_out : str | list | Irreps
        Input / output irrep specifications.
    bias : bool
        If ``True``, add a learnable bias to **all** ``m = 0`` output
        channels (invariant under rotations about the chosen axis).
        Default: ``True``.
    internal : bool
        If ``True`` (default), weights and bias are stored as learnable
        ``nn.Parameter``.  If ``False``, no parameters are created —
        call ``forward(x, weight=..., bias=...)`` with external tensors.
        Query ``weight_numel`` and ``bias_numel`` for required sizes.
    """

    def __init__(
        self,
        irreps_in: Union[str, list, Irreps],
        irreps_out: Union[str, list, Irreps],
        *,
        bias: bool = True,
        internal: bool = True,
    ):
        super().__init__()
        if not isinstance(irreps_in, Irreps):
            irreps_in = Irreps(irreps_in)
        if not isinstance(irreps_out, Irreps):
            irreps_out = Irreps(irreps_out)

        _check_natural_parity(irreps_in, irreps_out, "O2Linear")

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.internal = internal

        lmax = max(irreps_in.lmax, irreps_out.lmax) if len(irreps_in) and len(irreps_out) else 0

        def _collect_m_channels(irreps: Irreps):
            """Return {|m|: list_of_flat_channel_indices} for +m and -m."""
            pos = {}
            slices = irreps.slices()
            for i, (mult, ir) in enumerate(irreps):
                base = slices[i].start
                stride = ir.dim
                for am in range(ir.l + 1):
                    if am not in pos:
                        pos[am] = []
                    for k in range(mult):
                        offset = base + k * stride
                        if am == 0:
                            pos[am].append((offset + ir.l,))
                        else:
                            pos[am].append((offset + am + ir.l, offset + -am + ir.l))
            return pos

        in_ch = _collect_m_channels(irreps_in)
        out_ch = _collect_m_channels(irreps_out)

        # Build flat weight vector and index pattern
        weight_pieces: List[Tensor] = []
        row_indices: List[Tensor] = []
        col_indices: List[Tensor] = []
        repeats: List[int] = []
        bias_indices: List[int] = []
        block_info: List[tuple] = []  # (am, mo, mi) for non-empty blocks

        # Per-|m|-block metadata for the block-matmul forward:
        #   (am, mo, mi, n_comp, w_off, in_off, out_off)
        # with in-channel indices stored contiguously in _block_in_idx
        # (total length Σ mi*n_comp) and output indices in _block_out_idx
        # (total length Σ mo*n_comp).
        per_block: List[Tuple[int, int, int, int, int, int, int]] = []
        in_idx_parts: List[Tensor] = []
        out_idx_parts: List[Tensor] = []
        w_running = 0
        in_running = 0
        out_running = 0

        for am in range(lmax + 1):
            in_list = in_ch.get(am, [])
            out_list = out_ch.get(am, [])
            mi = len(in_list)
            mo = len(out_list)
            if mi == 0 or mo == 0:
                if am == 0 and bias and mo > 0:
                    for tup in out_list:
                        bias_indices.append(tup[0])
                continue

            block_info.append((am, mo, mi))
            w = torch.empty(mo, mi)
            nn.init.normal_(w, std=1.0 / math.sqrt(mi))
            weight_pieces.append(w.reshape(-1))

            n_comp = 1 if am == 0 else 2

            out_flat = torch.tensor(out_list, dtype=torch.long)  # (mo, n_comp)
            in_flat = torch.tensor(in_list, dtype=torch.long)     # (mi, n_comp)

            rows_parts = []
            cols_parts = []
            for c in range(n_comp):
                rows_parts.append(out_flat[:, c].unsqueeze(1).expand(mo, mi))
                cols_parts.append(in_flat[:, c].unsqueeze(0).expand(mo, mi))
            row_indices.append(torch.stack(rows_parts, dim=-1).reshape(-1))
            col_indices.append(torch.stack(cols_parts, dim=-1).reshape(-1))

            repeats.extend([n_comp] * (mo * mi))

            if am == 0 and bias:
                for tup in out_list:
                    bias_indices.append(tup[0])

            # Per-block gather/scatter indices (mi * n_comp / mo * n_comp long)
            in_flat_flat = in_flat.reshape(-1)       # (mi * n_comp,)
            out_flat_flat = out_flat.reshape(-1)     # (mo * n_comp,)
            in_idx_parts.append(in_flat_flat)
            out_idx_parts.append(out_flat_flat)

            per_block.append((am, mo, mi, n_comp, w_running, in_running, out_running))
            w_running += mo * mi
            in_running += mi * n_comp
            out_running += mo * n_comp

        # Weight numel (always computed)
        self._weight_numel = sum(p.numel() for p in weight_pieces) if weight_pieces else 0
        self._block_info = block_info

        # One dense (mo, mi) block per |m| sector.  Rebuilt from weight_pieces
        # (which are those same matrices, flattened) so the block order is the
        # flat order by construction -- see _FlatWeightCompat.
        _mats = [piece.reshape(int(mo), int(mi))
                 for piece, (_am, mo, mi) in zip(weight_pieces, block_info)]
        if internal:
            self.weight_blocks = nn.ParameterList(
                [nn.Parameter(m) for m in _mats])
        else:
            self.weight_blocks = None
        self._register_flat_weight_compat([tuple(m.shape) for m in _mats])

        # Index buffers are non-persistent (rebuilt here; see
        # utils.register_index_buffer).  _row_idx / _col_idx / _repeats /
        # _expand_idx describe the dense expansion (tests/analysis only).
        if row_indices:
            register_index_buffer(self, "_row_idx", torch.cat(row_indices))
            register_index_buffer(self, "_col_idx", torch.cat(col_indices))
        else:
            register_index_buffer(self, "_row_idx", torch.empty(0, dtype=torch.long))
            register_index_buffer(self, "_col_idx", torch.empty(0, dtype=torch.long))

        # Per-|m|-block gather/scatter indices for the block-matmul forward.
        if in_idx_parts:
            register_index_buffer(self, "_block_in_idx", torch.cat(in_idx_parts))
            register_index_buffer(self, "_block_out_idx", torch.cat(out_idx_parts))
        else:
            register_index_buffer(self, "_block_in_idx", torch.empty(0, dtype=torch.long))
            register_index_buffer(self, "_block_out_idx", torch.empty(0, dtype=torch.long))

        # Parallel List[int] metadata for the JIT-scripted forward.
        # (TorchScript cannot consume list-of-tuples; split into one list
        # per field, in the same iteration order as _block_in_idx /
        # _block_out_idx.)
        self._blk_mo: List[int] = [int(t[1]) for t in per_block]
        self._blk_mi: List[int] = [int(t[2]) for t in per_block]
        self._blk_ncomp: List[int] = [int(t[3]) for t in per_block]
        self._blk_w_off: List[int] = [int(t[4]) for t in per_block]
        self._blk_in_off: List[int] = [int(t[5]) for t in per_block]
        self._blk_out_off: List[int] = [int(t[6]) for t in per_block]
        self._per_block = per_block  # kept for backward/tests compat

        if repeats:
            register_index_buffer(self, "_repeats", torch.tensor(repeats, dtype=torch.long))
        else:
            register_index_buffer(self, "_repeats", torch.empty(0, dtype=torch.long))

        # Precompute expand-index for batched weight application:
        # expand_idx[j] = which flat-weight entry feeds expanded position j.
        # Equivalent to repeat_interleave but usable as a gather index for
        # batched weight tensors: expanded = w[..., _expand_idx].
        if self._weight_numel > 0 and repeats:
            _expand_idx = torch.arange(self._weight_numel).repeat_interleave(
                self._repeats
            )
            register_index_buffer(self, "_expand_idx", _expand_idx)
        else:
            register_index_buffer(self, "_expand_idx", torch.empty(0, dtype=torch.long))

        # Bias
        self._bias_numel = len(bias_indices) if bias else 0
        if bias and bias_indices:
            register_index_buffer(self, "_bias_idx",
                                  torch.tensor(bias_indices, dtype=torch.long))
            if internal:
                self.bias = nn.Parameter(torch.zeros(len(bias_indices)))
            else:
                self.bias = None
        else:
            self._bias_idx = None
            self.bias = None

        self._in_dim = irreps_in.dim
        self._out_dim = irreps_out.dim

    @property
    def weight_numel(self) -> int:
        """Number of weight scalars needed for the equivariant projection."""
        return self._weight_numel

    @property
    def bias_numel(self) -> int:
        """Number of bias scalars (0 if bias is disabled)."""
        return self._bias_numel

    def forward(
        self,
        x: O3Tensor,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        axis: Optional[Tensor] = None,
        D_cache_to_z: Optional[Dict] = None,
        D_cache_from_z: Optional[Dict] = None,
        parity_to_z: Optional[Tensor] = None,
        parity_from_z: Optional[Tensor] = None,
    ) -> O3Tensor:
        """Apply O(2) equivariant linear projection.

        Parameters
        ----------
        x : O3Tensor
            Input tensor with ``x.irreps == self.irreps_in``.
        weight : Tensor, optional
            Flat weight vector of shape ``(..., weight_numel)``.  Required
            when ``internal=False``; if provided when ``internal=True`` it
            overrides the stored parameters.  Leading batch dims are
            broadcast with ``x``.
        bias : Tensor, optional
            Bias vector of shape ``(..., bias_numel)``.  Same semantics
            as *weight*.
        axis : Tensor, optional
            Symmetry axis, shape ``(*, 3)``.  Supports per-sample /
            batched axes.  The vectors need **not** be unit-length —
            they are normalised internally.  When ``None`` (default),
            the +z axis is assumed and no rotation is performed.
        D_cache_to_z : dict, optional
            Pre-computed Wigner D-matrices ``{l: Tensor}`` for the
            rotation *to* the z-frame.  Skips recomputation when given.
        D_cache_from_z : dict, optional
            Pre-computed Wigner D-matrices for the rotation *from* the
            z-frame back to the original frame.
        parity_to_z : Tensor, optional
            Parity for R_to_z.
        parity_from_z : Tensor, optional
            Parity for R_from_z.
        """
        if x.irreps != self.irreps_in:
            raise ValueError(
                f"Expected input irreps {self.irreps_in}, got {x.irreps}"
            )

        # Resolve weight
        if weight is None and self.weight_blocks is None:
            raise ValueError(
                "No weight provided and internal=False — pass weight to forward()"
            )

        # Resolve bias
        b = bias if bias is not None else self.bias

        raw = x.tensor

        # ----- path with axis rotation ----------------------------------------
        if axis is not None:
            from .wigner import rotate_irreps

            # Skip rotation matrix computation when D caches are provided
            if D_cache_to_z is None or D_cache_from_z is None:
                from .utils import batched_rotation_to_z
                direction = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                R_to_z, R_from_z = batched_rotation_to_z(direction)

            # Rotate into z-frame (block-wise, no dense D matrix)
            raw = rotate_irreps(
                self.irreps_in, R_to_z if D_cache_to_z is None else None, raw,
                D_cache=D_cache_to_z, parity=parity_to_z,
            )

            out = (self._block_matmul(raw, weight) if weight is not None
                   else self._block_matmul_blocks(raw))

            # Bias
            if b is not None and self._bias_idx is not None:
                out[..., self._bias_idx] = out[..., self._bias_idx] + b

            # Rotate back from z-frame (block-wise, no dense D matrix)
            out = rotate_irreps(
                self.irreps_out,
                R_from_z if D_cache_from_z is None else None,
                out,
                D_cache=D_cache_from_z, parity=parity_from_z,
            )

            return O3Tensor(out, self.irreps_out)

        # ----- z-axis path (no rotation) --------------------------------------

        out = (self._block_matmul(raw, weight) if weight is not None
               else self._block_matmul_blocks(raw))

        # Bias on m=0 channels
        if b is not None and self._bias_idx is not None:
            out[..., self._bias_idx] = out[..., self._bias_idx] + b

        return O3Tensor(out, self.irreps_out)

    def _block_matmul(self, raw: Tensor, w: Tensor) -> Tensor:
        """Per-|m|-block equivariant matmul (scripted cores).

        Dispatches to :func:`_o2_linear_block_internal_core` when ``w``
        is a flat 1-D tensor and to :func:`_o2_linear_block_external_core`
        when ``w`` carries a per-edge batch dimension.  Both cores are
        ``@torch.jit.script`` decorated.
        """
        if w.dim() == 1:
            return _o2_linear_block_internal_core(
                raw, w,
                self._block_in_idx, self._block_out_idx,
                self._blk_mo, self._blk_mi, self._blk_ncomp,
                self._blk_w_off, self._blk_in_off, self._blk_out_off,
                self._out_dim,
            )
        return _o2_linear_block_external_core(
            raw, w,
            self._block_in_idx, self._block_out_idx,
            self._blk_mo, self._blk_mi, self._blk_ncomp,
            self._blk_w_off, self._blk_in_off, self._blk_out_off,
            self._out_dim,
        )

    def _block_matmul_blocks(self, raw: Tensor) -> Tensor:
        """Internal path: per-|m| block parameters, no flat vector involved."""
        return _o2_linear_block_internal_core_blocks(
            raw, list(self.weight_blocks), self._blk_mo, self._blk_mi,
            self._block_in_idx, self._block_out_idx,
            self._blk_ncomp, self._blk_in_off, self._blk_out_off,
            self._out_dim,
        )

    def extra_repr(self) -> str:
        return (
            f"irreps_in={self.irreps_in}, "
            f"irreps_out={self.irreps_out}, "
            f"bias={'m=0' if self._bias_numel > 0 else 'none'}, "
            f"internal={self.internal}, "
            f"weight_numel={self._weight_numel}, "
            f"bias_numel={self._bias_numel}"
        )


# ---------------------------------------------------------------------------
# SO(2) equivariant linear — JIT kernels
# ---------------------------------------------------------------------------


@torch.jit.script
def _so2_linear_block_internal_core(
    raw: Tensor,
    w: Tensor,
    block_in_idx: Tensor,
    block_out_idx: Tensor,
    mo_list: List[int],
    mi_list: List[int],
    ncomp_list: List[int],
    w_off_list: List[int],
    in_off_list: List[int],
    out_off_list: List[int],
    out_dim: int,
) -> Tensor:
    """Per-|m|-block matmul with SO(2) constraint (1-D internal weights).

    m=0 blocks use a standard (mo, mi) matmul (same as O(2)).
    |m|>0 blocks use a complex-linear structure with two weight matrices
    W_a and W_b:  y_{+m} = W_a x_{+m} - W_b x_{-m},
                  y_{-m} = W_b x_{+m} + W_a x_{-m}.
    """
    batch_shape = raw.shape[:-1]
    out = raw.new_zeros(batch_shape + (out_dim,))
    n_blocks = len(mo_list)
    for i in range(n_blocks):
        mo = mo_list[i]
        mi = mi_list[i]
        nc = ncomp_list[i]
        wo = w_off_list[i]
        io = in_off_list[i]
        oo = out_off_list[i]

        in_idx = block_in_idx.narrow(0, io, mi * nc)
        out_idx = block_out_idx.narrow(0, oo, mo * nc)

        x_blk = raw.index_select(-1, in_idx).reshape(batch_shape + (mi, nc))

        if nc == 1:
            W_m = w.narrow(0, wo, mo * mi).view(mo, mi)
            y_blk = torch.matmul(W_m, x_blk)
        else:
            W_a = w.narrow(0, wo, mo * mi).view(mo, mi)
            W_b = w.narrow(0, wo + mo * mi, mo * mi).view(mo, mi)
            x_p = x_blk.narrow(-1, 0, 1)
            x_n = x_blk.narrow(-1, 1, 1)
            y_p = torch.matmul(W_a, x_p) - torch.matmul(W_b, x_n)
            y_n = torch.matmul(W_b, x_p) + torch.matmul(W_a, x_n)
            y_blk = torch.cat([y_p, y_n], dim=-1)

        out.index_copy_(-1, out_idx, y_blk.reshape(batch_shape + (mo * nc,)))
    return out


@torch.jit.script
def _so2_linear_block_external_core(
    raw: Tensor,
    w: Tensor,
    block_in_idx: Tensor,
    block_out_idx: Tensor,
    mo_list: List[int],
    mi_list: List[int],
    ncomp_list: List[int],
    w_off_list: List[int],
    in_off_list: List[int],
    out_off_list: List[int],
    out_dim: int,
) -> Tensor:
    """Per-|m|-block batched matmul with SO(2) constraint (per-edge external weights)."""
    batch_shape = raw.shape[:-1]
    out = raw.new_zeros(batch_shape + (out_dim,))
    n_blocks = len(mo_list)
    for i in range(n_blocks):
        mo = mo_list[i]
        mi = mi_list[i]
        nc = ncomp_list[i]
        wo = w_off_list[i]
        io = in_off_list[i]
        oo = out_off_list[i]

        in_idx = block_in_idx.narrow(0, io, mi * nc)
        out_idx = block_out_idx.narrow(0, oo, mo * nc)

        x_blk = raw.index_select(-1, in_idx).reshape(batch_shape + (mi, nc))

        if nc == 1:
            W_m = w.narrow(-1, wo, mo * mi).reshape(batch_shape + (mo, mi))
            y_blk = torch.matmul(W_m, x_blk)
        else:
            W_a = w.narrow(-1, wo, mo * mi).reshape(batch_shape + (mo, mi))
            W_b = w.narrow(-1, wo + mo * mi, mo * mi).reshape(batch_shape + (mo, mi))
            x_p = x_blk.narrow(-1, 0, 1)
            x_n = x_blk.narrow(-1, 1, 1)
            y_p = torch.matmul(W_a, x_p) - torch.matmul(W_b, x_n)
            y_n = torch.matmul(W_b, x_p) + torch.matmul(W_a, x_n)
            y_blk = torch.cat([y_p, y_n], dim=-1)

        out.index_copy_(-1, out_idx, y_blk.reshape(batch_shape + (mo * nc,)))
    return out


# ---------------------------------------------------------------------------
# SO(2) equivariant linear
# ---------------------------------------------------------------------------


@torch.jit.script
def _so2_linear_block_internal_core_blocks(
    raw: Tensor,
    ws: List[Tensor],
    mo_list: List[int],
    mi_list: List[int],
    block_in_idx: Tensor,
    block_out_idx: Tensor,
    ncomp_list: List[int],
    in_off_list: List[int],
    out_off_list: List[int],
    out_dim: int,
) -> Tensor:
    """SO(2) per-|m|-block matmul reading BLOCK parameters directly.

    ``ws[i]`` is ``(mo, mi)`` for the m=0 block and ``(2, mo, mi)`` for every
    |m|>0 block, where ``[0]`` is W_a and ``[1]`` is W_b -- the real and
    imaginary parts of ONE complex operator ``[[W_a, -W_b], [W_b, W_a]]``.
    """
    batch_shape = raw.shape[:-1]
    out = raw.new_zeros(batch_shape + (out_dim,))
    n_blocks = len(ncomp_list)
    for i in range(n_blocks):
        W = ws[i]
        nc = ncomp_list[i]
        mo = mo_list[i]  # Python ints, not W.size(): symbolic under make_fx (compile ~10x slower)
        mi = mi_list[i]
        in_idx = block_in_idx.narrow(0, in_off_list[i], mi * nc)
        out_idx = block_out_idx.narrow(0, out_off_list[i], mo * nc)
        x_blk = raw.index_select(-1, in_idx).reshape(batch_shape + (mi, nc))
        if nc == 1:
            y_blk = torch.matmul(W, x_blk)
        else:
            W_a = W[0]
            W_b = W[1]
            x_p = x_blk.narrow(-1, 0, 1)
            x_n = x_blk.narrow(-1, 1, 1)
            y_p = torch.matmul(W_a, x_p) - torch.matmul(W_b, x_n)
            y_n = torch.matmul(W_b, x_p) + torch.matmul(W_a, x_n)
            y_blk = torch.cat([y_p, y_n], dim=-1)
        out.index_copy_(-1, out_idx, y_blk.reshape(batch_shape + (mo * nc,)))
    return out


class SO2Linear(_FlatWeightCompat, nn.Module):
    r"""SO(2) equivariant linear projection between two O3Tensor layouts.

    Identical to :class:`O2Linear` except for the constraint on
    ``|m| > 0`` blocks.  O(2) requires the *same* weight matrix for
    the ``+m`` and ``-m`` components (``W ⊗ I₂``).  SO(2) — which
    drops the reflection constraint — allows a **complex-linear**
    structure with *two* independent weight matrices per ``|m|`` block:

    .. math::

        \begin{pmatrix} y_{+m} \\ y_{-m} \end{pmatrix}
        = \begin{pmatrix} W_a & -W_b \\ W_b & W_a \end{pmatrix}
        \begin{pmatrix} x_{+m} \\ x_{-m} \end{pmatrix}

    This doubles the parameter count for ``|m| > 0`` blocks relative
    to :class:`O2Linear`, giving strictly more expressivity (equivalent
    to eSCN Eq. 20 rather than Eq. 14).

    The ``m = 0`` block is identical to :class:`O2Linear`.
    Like :class:`O2Linear` it requires natural-parity irreps.

    Parameters
    ----------
    irreps_in, irreps_out : str | list | Irreps
        Input / output irrep specifications.
    bias : bool
        If ``True``, add a learnable bias to **all** ``m = 0`` output
        channels (invariant under rotations about the chosen axis).
        Default: ``True``.
    internal : bool
        If ``True`` (default), weights and bias are stored as learnable
        ``nn.Parameter``.  If ``False``, no parameters are created —
        call ``forward(x, weight=..., bias=...)`` with external tensors.
        Query ``weight_numel`` and ``bias_numel`` for required sizes.
    """

    def __init__(
        self,
        irreps_in: Union[str, list, Irreps],
        irreps_out: Union[str, list, Irreps],
        *,
        bias: bool = True,
        internal: bool = True,
    ):
        super().__init__()
        if not isinstance(irreps_in, Irreps):
            irreps_in = Irreps(irreps_in)
        if not isinstance(irreps_out, Irreps):
            irreps_out = Irreps(irreps_out)

        _check_natural_parity(irreps_in, irreps_out, "SO2Linear")

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.internal = internal

        lmax = max(irreps_in.lmax, irreps_out.lmax) if len(irreps_in) and len(irreps_out) else 0

        def _collect_m_channels(irreps: Irreps):
            """Return {|m|: list_of_flat_channel_indices} for +m and -m."""
            pos = {}
            slices = irreps.slices()
            for i, (mult, ir) in enumerate(irreps):
                base = slices[i].start
                stride = ir.dim
                for am in range(ir.l + 1):
                    if am not in pos:
                        pos[am] = []
                    for k in range(mult):
                        offset = base + k * stride
                        if am == 0:
                            pos[am].append((offset + ir.l,))
                        else:
                            pos[am].append((offset + am + ir.l, offset + -am + ir.l))
            return pos

        in_ch = _collect_m_channels(irreps_in)
        out_ch = _collect_m_channels(irreps_out)

        weight_pieces: List[Tensor] = []
        bias_indices: List[int] = []
        block_info: List[tuple] = []

        per_block: List[Tuple[int, int, int, int, int, int, int]] = []
        in_idx_parts: List[Tensor] = []
        out_idx_parts: List[Tensor] = []
        w_running = 0
        in_running = 0
        out_running = 0

        for am in range(lmax + 1):
            in_list = in_ch.get(am, [])
            out_list = out_ch.get(am, [])
            mi = len(in_list)
            mo = len(out_list)
            if mi == 0 or mo == 0:
                if am == 0 and bias and mo > 0:
                    for tup in out_list:
                        bias_indices.append(tup[0])
                continue

            block_info.append((am, mo, mi))

            if am == 0:
                w = torch.empty(mo, mi)
                nn.init.normal_(w, std=1.0 / math.sqrt(mi))
                weight_pieces.append(w.reshape(-1))
                w_size = mo * mi
            else:
                w_a = torch.empty(mo, mi)
                nn.init.normal_(w_a, std=1.0 / math.sqrt(mi))
                w_b = torch.zeros(mo, mi)
                weight_pieces.append(w_a.reshape(-1))
                weight_pieces.append(w_b.reshape(-1))
                w_size = 2 * mo * mi

            n_comp = 1 if am == 0 else 2

            out_flat = torch.tensor(out_list, dtype=torch.long)
            in_flat = torch.tensor(in_list, dtype=torch.long)

            in_flat_flat = in_flat.reshape(-1)
            out_flat_flat = out_flat.reshape(-1)
            in_idx_parts.append(in_flat_flat)
            out_idx_parts.append(out_flat_flat)

            per_block.append((am, mo, mi, n_comp, w_running, in_running, out_running))
            w_running += w_size
            in_running += mi * n_comp
            out_running += mo * n_comp

            if am == 0 and bias:
                for tup in out_list:
                    bias_indices.append(tup[0])

        self._weight_numel = sum(p.numel() for p in weight_pieces) if weight_pieces else 0
        self._block_info = block_info

        # |m|=0 -> one dense (mo, mi) block.  |m|>0 -> a STACKED (2, mo, mi)
        # block holding (W_a, W_b).  Those are the real and imaginary parts of a
        # single complex operator, and stacking them is what lets a matrix-aware
        # optimiser treat the pair as ONE unit rather than two
        # independent matrices.  torch.stack([a, b]).reshape(-1) is identical to
        # the old cat(a.flat, b.flat), so the flat .weight view and the
        # legacy-checkpoint split hook stay byte-consistent with old files.
        _mats = []
        _pi = 0
        for _am, _mo, _mi in block_info:
            _mo, _mi = int(_mo), int(_mi)
            if int(_am) == 0:
                _mats.append(weight_pieces[_pi].reshape(_mo, _mi))
                _pi += 1
            else:
                _a = weight_pieces[_pi].reshape(_mo, _mi)
                _b = weight_pieces[_pi + 1].reshape(_mo, _mi)
                _mats.append(torch.stack([_a, _b]))
                _pi += 2
        if internal:
            self.weight_blocks = nn.ParameterList(
                [nn.Parameter(m) for m in _mats])
        else:
            self.weight_blocks = None
        self._register_flat_weight_compat([tuple(m.shape) for m in _mats])

        if in_idx_parts:
            register_index_buffer(self, "_block_in_idx", torch.cat(in_idx_parts))
            register_index_buffer(self, "_block_out_idx", torch.cat(out_idx_parts))
        else:
            register_index_buffer(self, "_block_in_idx", torch.empty(0, dtype=torch.long))
            register_index_buffer(self, "_block_out_idx", torch.empty(0, dtype=torch.long))

        self._blk_mo: List[int] = [int(t[1]) for t in per_block]
        self._blk_mi: List[int] = [int(t[2]) for t in per_block]
        self._blk_ncomp: List[int] = [int(t[3]) for t in per_block]
        self._blk_w_off: List[int] = [int(t[4]) for t in per_block]
        self._blk_in_off: List[int] = [int(t[5]) for t in per_block]
        self._blk_out_off: List[int] = [int(t[6]) for t in per_block]
        self._per_block = per_block

        self._bias_numel = len(bias_indices) if bias else 0
        if bias and bias_indices:
            register_index_buffer(self, "_bias_idx",
                                  torch.tensor(bias_indices, dtype=torch.long))
            if internal:
                self.bias = nn.Parameter(torch.zeros(len(bias_indices)))
            else:
                self.bias = None
        else:
            self._bias_idx = None
            self.bias = None

        self._in_dim = irreps_in.dim
        self._out_dim = irreps_out.dim

    @property
    def weight_numel(self) -> int:
        """Number of weight scalars needed for the equivariant projection."""
        return self._weight_numel

    @property
    def bias_numel(self) -> int:
        """Number of bias scalars (0 if bias is disabled)."""
        return self._bias_numel

    def forward(
        self,
        x: O3Tensor,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        axis: Optional[Tensor] = None,
        D_cache_to_z: Optional[Dict] = None,
        D_cache_from_z: Optional[Dict] = None,
        parity_to_z: Optional[Tensor] = None,
        parity_from_z: Optional[Tensor] = None,
    ) -> O3Tensor:
        """Apply SO(2) equivariant linear projection.

        Parameters are identical to :meth:`O2Linear.forward`.
        """
        if x.irreps != self.irreps_in:
            raise ValueError(
                f"Expected input irreps {self.irreps_in}, got {x.irreps}"
            )

        if weight is None and self.weight_blocks is None:
            raise ValueError(
                "No weight provided and internal=False — pass weight to forward()"
            )

        b = bias if bias is not None else self.bias

        raw = x.tensor

        if axis is not None:
            from .wigner import rotate_irreps

            if D_cache_to_z is None or D_cache_from_z is None:
                from .utils import batched_rotation_to_z
                direction = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                R_to_z, R_from_z = batched_rotation_to_z(direction)

            raw = rotate_irreps(
                self.irreps_in, R_to_z if D_cache_to_z is None else None, raw,
                D_cache=D_cache_to_z, parity=parity_to_z,
            )

            out = (self._block_matmul(raw, weight) if weight is not None
                   else self._block_matmul_blocks(raw))

            if b is not None and self._bias_idx is not None:
                out[..., self._bias_idx] = out[..., self._bias_idx] + b

            out = rotate_irreps(
                self.irreps_out,
                R_from_z if D_cache_from_z is None else None,
                out,
                D_cache=D_cache_from_z, parity=parity_from_z,
            )

            return O3Tensor(out, self.irreps_out)

        out = (self._block_matmul(raw, weight) if weight is not None
               else self._block_matmul_blocks(raw))

        if b is not None and self._bias_idx is not None:
            out[..., self._bias_idx] = out[..., self._bias_idx] + b

        return O3Tensor(out, self.irreps_out)

    def _block_matmul(self, raw: Tensor, w: Tensor) -> Tensor:
        """Per-|m|-block SO(2) equivariant matmul."""
        if w.dim() == 1:
            return _so2_linear_block_internal_core(
                raw, w,
                self._block_in_idx, self._block_out_idx,
                self._blk_mo, self._blk_mi, self._blk_ncomp,
                self._blk_w_off, self._blk_in_off, self._blk_out_off,
                self._out_dim,
            )
        return _so2_linear_block_external_core(
            raw, w,
            self._block_in_idx, self._block_out_idx,
            self._blk_mo, self._blk_mi, self._blk_ncomp,
            self._blk_w_off, self._blk_in_off, self._blk_out_off,
            self._out_dim,
        )

    def _block_matmul_blocks(self, raw: Tensor) -> Tensor:
        """Internal path: per-|m| block parameters, no flat vector involved."""
        return _so2_linear_block_internal_core_blocks(
            raw, list(self.weight_blocks), self._blk_mo, self._blk_mi,
            self._block_in_idx, self._block_out_idx,
            self._blk_ncomp, self._blk_in_off, self._blk_out_off,
            self._out_dim,
        )

    def extra_repr(self) -> str:
        return (
            f"irreps_in={self.irreps_in}, "
            f"irreps_out={self.irreps_out}, "
            f"bias={'m=0' if self._bias_numel > 0 else 'none'}, "
            f"internal={self.internal}, "
            f"weight_numel={self._weight_numel}, "
            f"bias_numel={self._bias_numel}"
        )
