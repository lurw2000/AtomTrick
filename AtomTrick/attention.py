"""O(3)-equivariant multi-head attention and transformer block.

K and V are computed **per edge**: the (pre-normed) source-atom features are
rotated into the edge frame (inter-atomic direction -> +z), mapped by a
*learned* O(2)-equivariant linear layer (:class:`~AtomTrick.o3.linear.O2Linear`:
one weight matrix per |m|, shared by the +m/-m components), rotated back, and
multiplied by a distance-dependent gate -- one scalar per (degree l, channel),
broadcast over m -- produced by a bias-free MLP on an exponential-normal radial
basis with a cosine cutoff, so the gate vanishes at the cutoff.  The O2Linear weights are ordinary parameters; they are *not*
generated from the distance.  ``kv_mode="so2"`` swaps in the SO(2) control
(:class:`~AtomTrick.o3.linear.SO2Linear`) and ``kv_mode="o3"`` a rotation-free
node-level O3Linear (ablations).

Q is a node-level O3Linear projection.  Scores are the O(3)-invariant inner
product ``<q_i, k_ij> / sqrt(d_head)`` per head; ``attn_nonlin`` selects the
weighting applied to them: ``"identity"`` (the paper model: raw, unnormalised
scores) or ``"softmax"`` (normalised over the neighbours of each atom; ablation).
Both produce invariant scalar weights, so the weighted sum of values stays
equivariant.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .o3.irreps import Irreps
from .o3.tensor import O3Tensor
from .o3.linear import O3Linear, O2Linear, SO2Linear
from .o3.inner import o3_inner
from .o3.norm import O3LayerNorm, O3SharedLayerNorm
from .o3.activation import O3GatedActivation
from .o3.utils import register_index_buffer
from .o3.escn_rotation import escn_wigner_D_caches
from .radial import ExpNormalSmearing
from .attn_utils import _build_head_perm, sparse_softmax


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class O3MultiHeadAttention(nn.Module):
    """O(3)-equivariant multi-head attention with geometric edge injection.

    **Q** is a node-level O3Linear projection.

    **K** and **V** are computed *per edge* with O2Linear (``k_mode`` /
    ``v_mode``):

    * The O(2) symmetry axis is the unit direction from the source to the
      destination atom; features are rotated into that edge frame, mapped
      by the learned O2Linear and rotated back.
    * Distance enters through a radial gate: ``distance -> RBF -> MLP ->
      one scalar per (l, channel)`` (separate MLPs for K and V, one RBF
      module per layer), multiplied onto the rotated-back K / V.

    Direction and distance information thus enter while full O(3)
    equivariance is preserved.  Attention weights are computed from the
    invariant scores as selected by ``attn_nonlin`` (see module docstring).

    Parameters
    ----------
    irreps : str | list | Irreps
        Hidden irrep layout.  Every multiplicity must be divisible by
        ``n_heads``.
    n_heads : int
        Number of attention heads.
    num_radial_basis : int
        Number of exponential-normal radial basis functions.
    cutoff : float
        Radial cutoff distance.
    radial_hidden : int
        Hidden-layer width inside the radial MLPs.
    radial_mode : str
        ``"gate"`` — learned O2Linear weights with per-channel radial gating
        (cheap, total_mult-dim MLP output).  The only supported mode.
    radial_rank : int
        Unused (legacy kwarg retained for config compatibility).
    k_mode, v_mode : str
        ``"o2"`` (paper model), ``"so2"`` (SO(2) control) or ``"o3"``
        (rotation-free O3Linear).
    attn_nonlin : str
        Weighting of the invariant scores: ``"identity"`` (paper model) or
        ``"softmax"`` (ablation).
    """

    def __init__(self, irreps, n_heads, *,
                 num_radial_basis=16, cutoff=5.0, radial_hidden=64,
                 radial_mode="gate", radial_rank=8,
                 k_mode="o2", v_mode="o2", rbf_type="expnorm",
                 attn_nonlin="softmax"):
        super().__init__()
        if not isinstance(irreps, Irreps):
            irreps = Irreps(irreps)
        self.irreps = irreps
        self.n_heads = n_heads
        self.k_mode = k_mode
        self.v_mode = v_mode
        self.attn_nonlin = attn_nonlin
        if attn_nonlin not in ("identity", "softmax"):
            raise ValueError(f"attn_nonlin must be 'identity' or 'softmax', got {attn_nonlin!r}")
        if rbf_type != "expnorm":
            raise ValueError(f"rbf_type must be 'expnorm', got {rbf_type!r}")
        if k_mode == "o3" and v_mode == "o3":
            radial_mode = "gate"  # o3 mode always uses gate-style radial
        self.radial_mode = radial_mode
        _KEdgeLinear = SO2Linear if k_mode == "so2" else O2Linear
        _VEdgeLinear = SO2Linear if v_mode == "so2" else O2Linear
        self.rbf_type = rbf_type

        def _make_rbf():
            return ExpNormalSmearing(num_radial_basis, cutoff)

        for mult, ir in irreps:
            if mult % n_heads != 0:
                raise ValueError(
                    f"Multiplicity {mult} for {ir} is not divisible by n_heads={n_heads}"
                )

        self.irreps_head = Irreps([(m // n_heads, ir) for m, ir in irreps])
        self.d_per_head = self.irreps_head.dim
        self.scale = 1.0 / math.sqrt(self.d_per_head)

        # Q: node-level O3Linear
        self.W_q = O3Linear(irreps, irreps, bias=False)

        mlp_rbf_in = num_radial_basis

        if radial_mode == "gate":
            # Fixed weights + per-channel radial gating
            # K projection: O3Linear (no rotation) or O2/SO2Linear (rotation)
            if k_mode == "o3":
                self.o3_k = O3Linear(irreps, irreps, bias=False)
            else:
                self.o2_k = _KEdgeLinear(irreps, irreps, internal=True, bias=False)
            # V projection: O3Linear (no rotation) or O2/SO2Linear (rotation)
            if v_mode == "o3":
                self.o3_v = O3Linear(irreps, irreps, bias=False)
            else:
                self.o2_v = _VEdgeLinear(irreps, irreps, internal=True, bias=False)

            total_mult = sum(m for m, _ in irreps)
            self.rbf = _make_rbf()
            self.mlp_k = nn.Sequential(
                nn.Linear(mlp_rbf_in, radial_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(radial_hidden, total_mult, bias=False),
            )
            self.mlp_v = nn.Sequential(
                nn.Linear(mlp_rbf_in, radial_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(radial_hidden, total_mult, bias=False),
            )

            # Precompute expand index to map (*, total_mult) → (*, dim) via gather
            gate_repeats = []
            for mult, ir in irreps:
                gate_repeats.extend([ir.dim] * mult)
            _gate_expand_idx = torch.arange(total_mult).repeat_interleave(
                torch.tensor(gate_repeats, dtype=torch.long))
            register_index_buffer(self, "_gate_expand_idx", _gate_expand_idx)
        else:
            raise ValueError(f"Unknown radial_mode: {radial_mode!r}")

        # Output: node-level O3Linear
        self.W_o = O3Linear(irreps, irreps, bias=False)

        # Head-splitting permutations
        perm = _build_head_perm(irreps, n_heads)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm))
        register_index_buffer(self, "_head_perm", perm)
        register_index_buffer(self, "_head_inv_perm", inv_perm)

    # ----- head split / merge ------------------------------------------------

    def _split_heads(self, x_raw):
        """(*, D) irrep-major -> (*, H, D_head) head-major.

        Rewritten from fancy-index (``x[..., perm]``) to slice+view+cat.
        Numerically bit-exact with the original, but the backward path avoids
        ``aten::_index_put_impl_`` (the single biggest CUDA-time sink in the
        default-forward training profile). slice/cat have cheap backwards.
        """
        H = self.n_heads
        batch = x_raw.shape[:-1]
        parts = []
        for i, (mult, ir) in enumerate(self.irreps):
            sl = self.irreps._slices[i]
            mh = mult // H
            block = x_raw[..., sl.start:sl.stop]                # (*, mult*dim) slice (view)
            block = block.reshape(*batch, H, mh, ir.dim)        # (*, H, mh, dim)
            block = block.reshape(*batch, H, mh * ir.dim)       # (*, H, mh*dim)
            parts.append(block)
        return torch.cat(parts, dim=-1)                         # (*, H, d_per_head)

    def _merge_heads(self, x):
        """(*, H, D_head) head-major -> (*, D) irrep-major. Inverse of _split_heads."""
        H = self.n_heads
        batch = x.shape[:-2]
        parts = []
        offset = 0
        for mult, ir in self.irreps:
            mh = mult // H
            block = x[..., offset:offset + mh * ir.dim]          # (*, H, mh*dim)
            block = block.reshape(*batch, H, mh, ir.dim)         # (*, H, mh, dim)
            block = block.reshape(*batch, H * mh, ir.dim)        # (*, mult, dim)
            block = block.reshape(*batch, H * mh * ir.dim)       # (*, mult*dim)
            parts.append(block)
            offset += mh * ir.dim
        return torch.cat(parts, dim=-1)                          # (*, D)

    def _apply_gate(self, tensor, gate_raw):
        """Multiply ``tensor`` (E, D) by a per-irrep gate (E, total_mult).

        Uses ``torch.gather`` (whose backward is scatter_add) to expand the
        per-multiplicity gate to per-channel, then a single elementwise mul.
        Replaces both the old fancy-index path (``gate_raw[:, _gate_expand_idx]``,
        whose index_put backward is slow) and the earlier per-irrep Python loop
        (which had many small mul launches and blocked torch.compile tracing).
        Forward is bit-exact; gradient wrt gate matches within FP32 rounding.
        """
        idx = self._gate_expand_idx.expand(*tensor.shape[:-1], self._gate_expand_idx.shape[0])
        expanded_gate = gate_raw.gather(-1, idx)
        return tensor * expanded_gate

    # ----- forward ------------------------------------------------------------

    def forward(self, x, edge_index, R=None, edge_geom=None, edge_mask=None):
        """
        Parameters
        ----------
        x : O3Tensor, shape (N, D)
            Node features.
        edge_index : LongTensor, shape (2, E)
        R : Tensor, shape (N, 3), optional
            Atom positions.  Only used when ``edge_geom`` is omitted (standalone
            use of this block); the model always passes ``edge_geom``.
        edge_geom : dict, optional
            Pre-computed edge geometry (diff, dist, Wigner D caches).
            Computed once per forward by the model and shared across layers.
        edge_mask : BoolTensor, shape (E,), optional
            True for real edges, False for padding.  Padding edges are
            excluded from attention via ``-inf`` masking.

        Returns
        -------
        O3Tensor, shape (N, D)
        """
        src, dst = edge_index
        N = x.shape[0]

        # ---- edge geometry ---------------------------------------------------
        if edge_geom is not None:
            diff = edge_geom["diff"]
            dist = edge_geom["dist"]
            D_cache_to_z = edge_geom["D_cache_to_z"]
            D_cache_from_z = edge_geom["D_cache_from_z"]
            par_to_z = edge_geom["parity_to_z"]
            par_from_z = edge_geom["parity_from_z"]
        elif R is not None:
            # Standalone use (no model-level edge_geom): rebuild the geometry
            # here.  Non-periodic only — edge_shift lives in the model.
            diff = R[dst] - R[src]                                 # (E, 3)
            dist = diff.norm(dim=-1)                               # (E,)
            direction = diff / diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            ls = {ir.l for _, ir in self.irreps}
            D_cache_to_z, par_to_z, D_cache_from_z, par_from_z = escn_wigner_D_caches(
                direction, ls)
        else:
            raise ValueError("O3MultiHeadAttention.forward needs edge_geom or R")

        # ---- radial features ---------------------------------------------------
        rbf_feat = self.rbf(dist)                              # (E, B)

        # ---- Q: node-level O3Linear -----------------------------------------
        Q = self.W_q(x)                                        # O3Tensor (N, D)

        x_src = x[src]                                         # O3Tensor (E, D)

        if True:  # gate is the only radial_mode (full/lora/adapt removed)
            # Gate mode: fixed weights + per-channel radial gating.
            # The gate is produced per multiplicity (one scalar per irrep channel)
            # and broadcast across the (2l+1) magnetic components via _apply_gate
            # (slice+broadcast multiply) instead of the older fancy-index
            # expansion, which generated an expensive index_put backward.
            gate_k_raw = self.mlp_k(rbf_feat)                   # (E, total_mult)
            gate_v_raw = self.mlp_v(rbf_feat)                   # (E, total_mult)

            k_needs_rot = (self.k_mode != "o3")
            v_needs_rot = (self.v_mode != "o3")

            # Compute rotation to z-frame if needed by either K or V
            if k_needs_rot or v_needs_rot:
                from AtomTrick.o3.wigner import rotate_irreps
                x_src_z = rotate_irreps(
                    self.irreps, None, x_src.tensor,
                    D_cache=D_cache_to_z, parity=par_to_z,
                )

            # K projection
            if self.k_mode == "o3":
                K_raw = self.o3_k(x_src)
                K_edge = O3Tensor(self._apply_gate(K_raw.tensor, gate_k_raw), self.irreps)
            else:
                K_z = self.o2_k(O3Tensor(x_src_z, self.irreps))
                K_tensor = rotate_irreps(
                    self.irreps, None, K_z.tensor,
                    D_cache=D_cache_from_z, parity=par_from_z,
                )
                K_edge = O3Tensor(self._apply_gate(K_tensor, gate_k_raw), self.irreps)

            # V projection
            if self.v_mode == "o3":
                V_raw = self.o3_v(x_src)
                V_edge = O3Tensor(self._apply_gate(V_raw.tensor, gate_v_raw), self.irreps)
            else:
                V_z = self.o2_v(O3Tensor(x_src_z, self.irreps))
                V_tensor = rotate_irreps(
                    self.irreps, None, V_z.tensor,
                    D_cache=D_cache_from_z, parity=par_from_z,
                )
                V_edge = O3Tensor(self._apply_gate(V_tensor, gate_v_raw), self.irreps)

        # ---- multi-head attention -------------------------------------------
        Q_h = self._split_heads(Q.tensor)                      # (N, H, Dh)
        K_h = self._split_heads(K_edge.tensor)                 # (E, H, Dh)
        V_h = self._split_heads(V_edge.tensor)                 # (E, H, Dh)

        # Scores = scale·<Q_h[dst], K_h>.
        q_i = O3Tensor(Q_h[dst], self.irreps_head)         # (E, H, Dh)
        k_j = O3Tensor(K_h, self.irreps_head)              # (E, H, Dh)
        scores = o3_inner(q_i, k_j) * self.scale           # (E, H)
        # The normalised mode (softmax) needs padded neighbours pushed to -inf so
        # they drop out of the per-node normalisation; the unnormalised mode
        # (identity) is masked to zero on the resulting weights instead.
        _normed = self.attn_nonlin == "softmax"
        if edge_mask is not None and _normed:
            scores = scores.masked_fill(~edge_mask.unsqueeze(-1), -1e9)
        if self.attn_nonlin == "identity":
            alpha = scores                                       # (E, H)
        else:
            alpha = sparse_softmax(scores, dst, N)               # (E, H)
        if edge_mask is not None and not _normed:
            alpha = alpha.masked_fill(~edge_mask.unsqueeze(-1), 0.0)

        # out[n] = Σ_{e:dst=n} alpha[e,h]·V_h[e,h,:].  Eager weighted scatter.
        msg = alpha.unsqueeze(-1) * V_h                    # (E, H, Dh)
        out = torch.zeros(N, self.n_heads, self.d_per_head,
                          device=x.device, dtype=msg.dtype)
        out = out.scatter_add(0, dst[:, None, None].expand_as(msg), msg)

        out_flat = self._merge_heads(out)                       # (N, D)
        return self.W_o(O3Tensor(out_flat, self.irreps))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class O3TransformerBlock(nn.Module):
    """Pre-norm equivariant transformer block: Attention + FFN with residuals.

    Parameters
    ----------
    irreps : str | list | Irreps
        Hidden irrep layout.
    n_heads : int
        Number of attention heads.
    ff_mult : int
        Multiplicity expansion factor for the FFN intermediate layer.
    num_radial_basis : int
        Passed through to attention.
    cutoff : float
        Passed through to attention.
    radial_hidden : int
        Passed through to attention.
    """

    def __init__(self, irreps, n_heads, ff_mult=4, *,
                 num_radial_basis=16, cutoff=5.0, radial_hidden=64,
                 radial_mode="gate", radial_rank=8,
                 k_mode="o2", v_mode="o2",
                 ffn_act_mode="gate",
                 rbf_type="expnorm",
                 layernorm_mode="default",
                 attn_nonlin="softmax"):
        super().__init__()
        if not isinstance(irreps, Irreps):
            irreps = Irreps(irreps)
        _NormClass = O3SharedLayerNorm if layernorm_mode == "shared" else O3LayerNorm

        self.norm1 = _NormClass(irreps)
        self.attn = O3MultiHeadAttention(
            irreps, n_heads,
            num_radial_basis=num_radial_basis,
            cutoff=cutoff,
            radial_hidden=radial_hidden,
            radial_mode=radial_mode,
            radial_rank=radial_rank,
            k_mode=k_mode,
            v_mode=v_mode,
            rbf_type=rbf_type,
            attn_nonlin=attn_nonlin,
        )
        self.norm2 = _NormClass(irreps)

        # FFN: bias-free O3Linear up-projection (x ff_mult), activation, O3Linear down.
        act_irreps = Irreps([(m * ff_mult, ir) for m, ir in irreps])
        if ffn_act_mode != "gate":
            raise ValueError(f"ffn_act_mode must be 'gate', got {ffn_act_mode!r}")
        self.ffn_act = O3GatedActivation(act_irreps)
        self.ffn_up = O3Linear(irreps, self.ffn_act.irreps_in, bias=False)
        self.ffn_down = O3Linear(act_irreps, irreps, bias=False)

    def forward(self, x, edge_index, R=None, edge_geom=None, edge_mask=None):
        """
        Parameters
        ----------
        x : O3Tensor, shape (N, D)
        edge_index : LongTensor, shape (2, E)
        R : Tensor, shape (N, 3), optional
            Atom positions.  Only needed when ``edge_geom`` is omitted.
        edge_geom : dict, optional
            Pre-computed edge geometry from model forward.
        edge_mask : BoolTensor, shape (E,), optional
            True for real edges, False for padding.
        """
        x = x + self.attn(self.norm1(x), edge_index, R, edge_geom=edge_geom, edge_mask=edge_mask)
        h = self.ffn_down(self.ffn_act(self.ffn_up(self.norm2(x))))
        return x + h
