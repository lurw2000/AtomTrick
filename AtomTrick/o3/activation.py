"""Equivariant activations for O3Tensors."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .irreps import Irreps
from .tensor import O3Tensor
from .utils import register_index_buffer


# ── JIT-scripted gated-activation core (silu + sigmoid hardcoded) ───────────
# Most O3Attn configs use silu for scalars and sigmoid for gates (the e3nn
# defaults). Hardcoding these lets us script the full forward into a single
# fused TorchScript callable.

@torch.jit.script
def _o3_gated_silu_sigmoid_core(
    raw: Tensor,
    s_dim: int,
    g_dim: int,
    gate_expand_idx: Tensor,
) -> Tensor:
    """Gated activation with silu(scalars) + sigmoid(gates) baked in.

    Layout: raw[..., :s_dim] = scalar features, raw[..., s_dim:s_dim+g_dim] =
    gate scalars (one per non-scalar multiplicity channel), raw[..., s_dim+g_dim:]
    = non-scalar features. Gates are broadcast to non-scalar channels via
    ``gather`` (cheap scatter_add backward, no fancy index).
    """
    scalars = raw.narrow(-1, 0, s_dim)
    gates_raw = raw.narrow(-1, s_dim, g_dim)
    gated_raw = raw.narrow(-1, s_dim + g_dim, raw.size(-1) - s_dim - g_dim)

    scalars_out = F.silu(scalars)

    if g_dim > 0:
        gates = torch.sigmoid(gates_raw)
        idx_shape = gated_raw.shape[:-1] + (gate_expand_idx.shape[0],)
        idx = gate_expand_idx.expand(idx_shape)
        expanded = gates.gather(-1, idx)
        gated_out = gated_raw * expanded
        return torch.cat([scalars_out, gated_out], dim=-1)
    return scalars_out


# ── Loop-free norm-gated activation core (SiLU hot path) ────────────────────

class O3GatedActivation(nn.Module):
    """Gated equivariant activation for O3Tensors (e3nn-style Gate).

    * **Scalars (l=0, p=+1)**: pointwise nonlinearity (SiLU by default).
    * **Non-scalar channels**: multiplied by scalar gates passed through
      sigmoid.  One gate scalar per non-scalar multiplicity channel.

    The input must contain extra ``0e`` channels to serve as gates.
    Use :attr:`irreps_in` to determine the required input layout::

        irreps_in = [scalar_features] + [gate_scalars (0e)] + [non_scalar_features]

    Parameters
    ----------
    irreps_out : str | list | Irreps
        Desired output irreps (e.g. ``"64x0e + 64x1o + 64x2e"``).
    scalar_act : callable
        Activation for scalar channels (default: ``F.silu``).
    gate_act : callable
        Activation for gate scalars (default: ``torch.sigmoid``).
    """

    def __init__(self, irreps_out, scalar_act=F.silu, gate_act=torch.sigmoid):
        super().__init__()
        if not isinstance(irreps_out, Irreps):
            irreps_out = Irreps(irreps_out)

        self.irreps_out = irreps_out
        self.scalar_act = scalar_act
        self.gate_act = gate_act

        # Separate scalar and non-scalar entries
        scalar_entries = [(m, ir) for m, ir in irreps_out if ir.l == 0 and ir.p == 1]
        gated_entries = [(m, ir) for m, ir in irreps_out if not (ir.l == 0 and ir.p == 1)]

        self._n_scalar_features = sum(m for m, _ in scalar_entries)
        self._n_gate_scalars = sum(m for m, _ in gated_entries)
        self._gated_dims = [(m, ir.dim) for m, ir in gated_entries]

        # Input layout: [scalar_features (0e)] + [gate_scalars (0e)] + [non-scalars]
        from .irreps import Irrep
        gate_entries = [(self._n_gate_scalars, Irrep(0, 1))] if self._n_gate_scalars > 0 else []
        self.irreps_in = Irreps(scalar_entries + gate_entries + gated_entries)

        # Precompute gate-expand index so the forward uses torch.gather instead
        # of a Python for-loop over _gated_dims. gather's backward is a cheap
        # scatter_add (vs fancy-index's expensive index_put_).
        gate_expand: list = []
        g_off = 0
        for m, d in self._gated_dims:
            for gi in range(g_off, g_off + m):
                gate_expand.extend([gi] * d)
            g_off += m
        if gate_expand:
            register_index_buffer(
                self, "_gate_expand_idx",
                torch.tensor(gate_expand, dtype=torch.long),
            )
        else:
            register_index_buffer(
                self, "_gate_expand_idx",
                torch.empty(0, dtype=torch.long),
            )

    def forward(self, x):
        """Apply gated activation. Accepts O3Tensor or raw Tensor.

        Fast path: if scalar_act is silu and gate_act is sigmoid (the standard
        O3Attn / e3nn defaults), dispatches to a JIT-scripted core. Otherwise
        falls back to the eager implementation.
        """
        is_o3t = isinstance(x, O3Tensor)
        raw = x.tensor if is_o3t else x

        if self.scalar_act is F.silu and self.gate_act is torch.sigmoid:
            out = _o3_gated_silu_sigmoid_core(
                raw,
                self._n_scalar_features,
                self._n_gate_scalars,
                self._gate_expand_idx,
            )
        else:
            # Fallback eager path (rarely used)
            scalars = raw[..., :self._n_scalar_features]
            gates_raw = raw[..., self._n_scalar_features:self._n_scalar_features + self._n_gate_scalars]
            gated_raw = raw[..., self._n_scalar_features + self._n_gate_scalars:]
            scalars_out = self.scalar_act(scalars)
            if self._n_gate_scalars > 0:
                gates = self.gate_act(gates_raw)
                idx = self._gate_expand_idx.expand(*raw.shape[:-1], self._gate_expand_idx.shape[0])
                expanded_gates = gates.gather(-1, idx)
                gated_out = gated_raw * expanded_gates
                out = torch.cat([scalars_out, gated_out], dim=-1)
            else:
                out = scalars_out

        return O3Tensor(out, self.irreps_out) if is_o3t else out

    def extra_repr(self):
        s_name = getattr(self.scalar_act, "__name__", str(self.scalar_act))
        g_name = getattr(self.gate_act, "__name__", str(self.gate_act))
        return (f"irreps_in={self.irreps_in}, irreps_out={self.irreps_out}, "
                f"scalar_act={s_name}, gate_act={g_name}")
