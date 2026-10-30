"""AtomTrick: O(3)-equivariant graph transformer for energies and forces.

Hidden representations are O3Tensors (direct sums of natural-parity O(3)
irreps).  Atoms are embedded by atomic number into the scalar (0e) channels;
higher-degree channels start at zero.  Each of the ``n_layers`` pre-norm
transformer blocks (:class:`~AtomTrick.attention.O3TransformerBlock`) applies

* equivariant multi-head attention: node-level O3Linear queries; keys and
  values computed per edge by a *learned* O(2)-linear map in the edge frame
  (rotate -> O2Linear -> rotate back), times a distance-dependent radial gate
  (bias-free MLP on an exponential-normal radial basis, one scalar per (l,
  channel)); invariant scores ``<q_i, k_ij>/sqrt(d_head)`` weighted as set by
  ``attn_nonlin`` (``"identity"`` in the paper model), and
* an equivariant gated FFN,

each with a residual connection.  A final LayerNorm and an MLP on the
invariant (l=0) channels give per-atom energies, summed per molecule.  Forces
are ``F = -dE/dR`` by autograd.

The constructor defaults below are a small legacy configuration kept for API
compatibility; the paper configuration is the default of ``train.py`` (and
``configs/``).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .o3.irreps import Irreps
from .o3.tensor import O3Tensor
from .o3.linear import O3Linear
from .o3.norm import O3LayerNorm, O3SharedLayerNorm
from .o3.utils import register_index_buffer
from .o3.escn_rotation import escn_wigner_D_caches
from .attention import O3TransformerBlock


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AtomTrick(nn.Module):
    """O(3)-equivariant graph transformer for molecular property prediction.

    Atoms are embedded by atomic number into the scalar (0e) channels of
    an O3Tensor.  Geometric information (inter-atomic direction and
    distance) is injected per edge through learned O2Linear projections
    for K and V (applied in the edge frame) that are modulated by a
    radial gate network.

    Outputs per-atom forces (1o vector) and per-molecule energy (0e scalar).

    Parameters
    ----------
    irreps_hidden : str | list | Irreps
        Hidden irrep layout, e.g. ``"64x0e + 64x1o + 64x2e + 64x3o"``.
        Every multiplicity must be divisible by ``n_heads``, and every irrep
        must have natural parity ``p = (-1)**l`` (required by O2Linear).
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer blocks.
    ff_mult : int
        FFN intermediate multiplicity factor (transformer blocks).
    max_z : int
        Maximum atomic number for the embedding table.
    num_radial_basis : int
        Number of exponential-normal radial basis functions.
    cutoff : float
        Radial cutoff distance.
    radial_hidden : int
        Hidden-layer width inside the radial MLPs.
    energy_hidden : int
        Number of intermediate scalar channels in the energy head.
    radial_rank, energy_correlation : int
        Unused; accepted only so that old configs / call sites still work.
    escn_frame : str
        How the Wigner-D matrices of the eSCN edge frame are evaluated.
        ``"polefree"`` (default) keeps the Euler angles away from their poles,
        so forces are exactly equivariant for every edge direction.
        ``"euler"`` is the direct Euler-angle evaluation of the same frame,
        kept for the latency comparison of the two evaluations
        (``benchmarks/latency/``); its forces miss the frame's
        contribution for edges within ~1.5e-3 rad of the eSCN gimbal
        directions.  The two are the same function up to float round-off (see
        :func:`AtomTrick.o3.escn_rotation.escn_wigner_D_caches`).

    Buffers: ``atomic_ref_energy`` and ``energy_bias`` hold data-derived
    values and are saved in checkpoints.  All other buffers in the model
    are index tables rebuilt at construction and are NOT saved (older
    checkpoints that still contain them load fine; see
    :func:`AtomTrick.o3.utils.register_index_buffer`).
    """

    def __init__(self, irreps_hidden="16x0e + 16x1o + 16x2e",
                 n_heads=4, n_layers=4, ff_mult=4, max_z=100,
                 num_radial_basis=16, cutoff=5.0, radial_hidden=64,
                 energy_hidden=64, radial_mode="gate", radial_rank=8,
                 energy_correlation=2,
                 k_mode="o2", v_mode="o2",
                 ffn_act_mode="gate",
                 rbf_type="expnorm",
                 layernorm_mode="default",
                 attn_nonlin="softmax",
                 atomic_ref=None,
                 escn_frame="polefree"):
        super().__init__()
        if escn_frame not in ("polefree", "euler"):
            raise ValueError(f"escn_frame must be 'polefree' or 'euler', got {escn_frame!r}")
        self.escn_frame = escn_frame
        if not isinstance(irreps_hidden, Irreps):
            irreps_hidden = Irreps(irreps_hidden)
        self.irreps_hidden = irreps_hidden
        self.radial_mode = radial_mode
        self.layernorm_mode = layernorm_mode
        _NormClass = O3SharedLayerNorm if layernorm_mode == "shared" else O3LayerNorm

        # Number of scalar (0e) channels for the atom embedding
        n_scalar = sum(m for m, ir in irreps_hidden if ir.l == 0 and ir.p == 1)
        if n_scalar == 0:
            raise ValueError("irreps_hidden must contain at least one 0e sector")
        self.n_scalar = n_scalar
        self.atom_embed = nn.Embedding(max_z, n_scalar)

        # Precompute which flat indices correspond to the 0e channels
        scalar_ch = irreps_hidden._lp_ch_info.get((0, 1))
        if isinstance(scalar_ch, slice):
            self._scalar_slice = scalar_ch
        else:
            register_index_buffer(self, "_scalar_idx",
                                  torch.tensor(scalar_ch, dtype=torch.long))
            self._scalar_slice = None

        self.layers = nn.ModuleList([
            O3TransformerBlock(
                irreps_hidden, n_heads, ff_mult,
                num_radial_basis=num_radial_basis,
                cutoff=cutoff,
                radial_hidden=radial_hidden,
                radial_mode=radial_mode,
                radial_rank=radial_rank,
                k_mode=k_mode,
                v_mode=v_mode,
                ffn_act_mode=ffn_act_mode,
                rbf_type=rbf_type,
                layernorm_mode=layernorm_mode,
                attn_nonlin=attn_nonlin,
            )
            for _ in range(n_layers)
        ])

        self.out_norm = _NormClass(irreps_hidden)

        # Energy head: the l=0 (invariant) channels, then a bias-free MLP to one
        # energy per atom.
        self._compiled_energy = None  # set via compile_energy()

        self.energy_proj = nn.Sequential(
            nn.Linear(n_scalar, energy_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(energy_hidden, energy_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(energy_hidden, 1, bias=False),
        )

        # Atomic reference energies — added back to predictions at inference
        # so the model outputs total energies while training on interaction energies.
        from .constants import ATOMIC_REFERENCE_ENERGIES
        ref_source = atomic_ref if atomic_ref is not None else ATOMIC_REFERENCE_ENERGIES
        # JSON round-trips dict keys as strings; accept either int or str.
        ref_source = {int(k): float(v) for k, v in ref_source.items()}
        _ref = torch.zeros(max_z, dtype=torch.float32)
        for z, e in ref_source.items():
            if z < max_z:
                _ref[z] = e
        self.register_buffer("atomic_ref_energy", _ref)
        # Per-element energy offset.  Always zero (nothing in this code fits it);
        # kept, and applied in add_atomic_reference, so that checkpoints load strictly.
        self.register_buffer("energy_bias", torch.zeros(max_z, dtype=torch.float32))

        # Backwards compat: older checkpoints predate energy_bias.  Inject a
        # zero tensor at load time so strict=True load_state_dict still works.
        def _energy_bias_compat_hook(state_dict, prefix, *_args, **_kwargs):
            key = f"{prefix}energy_bias"
            if key not in state_dict:
                state_dict[key] = torch.zeros_like(self.energy_bias)
        self._register_load_state_dict_pre_hook(_energy_bias_compat_hook)

    def energy_bias_per_mol(self, Z, mol_idx, n_mol):
        """Per-molecule contribution of the fitted energy bias.

        Returns ``Σ_z energy_bias[z] · N_z`` per molecule — the calibration
        correction that must be added to the model's interaction-energy
        prediction before comparing against ``batch["E"]`` or converting
        to a total energy.
        """
        out = torch.zeros(n_mol, device=Z.device, dtype=self.energy_bias.dtype)
        out.scatter_add_(0, mol_idx, self.energy_bias[Z])
        return out

    def add_atomic_reference(self, E_pred, Z, mol_idx, n_mol):
        """Add atomic reference energies (+ fitted bias) back to E_pred.

        Parameters
        ----------
        E_pred : Tensor, shape (n_mol,)
            Predicted interaction energies.
        Z : Tensor, shape (total_atoms,)
            Atomic numbers.
        mol_idx : Tensor, shape (total_atoms,)
            Molecule index per atom.
        n_mol : int

        Returns
        -------
        Tensor, shape (n_mol,)
            Total energies (interaction + atomic reference + fitted bias).
        """
        ref = (self.atomic_ref_energy + self.energy_bias).to(E_pred.dtype)
        E_ref = torch.zeros(n_mol, device=E_pred.device, dtype=E_pred.dtype)
        E_ref.scatter_add_(0, mol_idx, ref[Z])
        return E_pred + E_ref

    def compile_energy(self, dynamic=False):
        """``torch.compile`` the energy function.

        Parameters
        ----------
        dynamic : bool
            If True, use dynamic shapes so varying atom/edge counts are
            handled without recompilation.
        """
        self._compiled_energy = torch.compile(self._energy, dynamic=dynamic)

    def _energy(self, R, Z, edge_index, mol_idx, n_mol,
                edge_shift=None, edge_mask=None):
        """Total energy + per-molecule energies from atom positions.

        ``R`` is the only geometric input, so ``autograd.grad(E_total, R)``
        in :meth:`forward` is the whole force computation.  One forward pass
        yields both the scalar E_total (differentiated for the forces) and the
        per-molecule E_pred (used by the energy loss).

        Returns ``(E_total, E_pred)`` where E_total is a scalar and
        E_pred has shape ``(n_mol,)``.
        """
        src, dst = edge_index
        diff = R[dst] - R[src]
        if edge_shift is not None:
            # Periodic (OC20): true minimum-image displacement.
            diff = diff + edge_shift
        dist = diff.norm(dim=-1)
        direction = diff / dist.unsqueeze(-1).clamp(min=1e-8)
        # eSCN-style decoupled-frame Wigner-D: pole-free at axis-aligned edges,
        # identical model function to the old rotate-to-lab-z construction
        # (validated: fp64 match 8.5e-13, O(3) equivariance to fp64 eps).
        ls = {ir.l for _, ir in self.irreps_hidden}
        D_cache_to_z, par_to_z, D_cache_from_z, par_from_z = escn_wigner_D_caches(
            direction, ls, pole_free=self.escn_frame == "polefree")
        N = Z.shape[0]
        if edge_mask is not None:
            deg = dist.new_zeros(N).scatter_add_(0, dst, edge_mask.to(dist.dtype))
        else:
            deg = dist.new_zeros(N).scatter_add_(0, dst, torch.ones_like(dst, dtype=dist.dtype))
        deg_inv_sqrt = deg.clamp(min=1).rsqrt().unsqueeze(-1)  # (N, 1)

        edge_geom = {
            "diff": diff, "dist": dist,
            "D_cache_to_z": D_cache_to_z, "D_cache_from_z": D_cache_from_z,
            "parity_to_z": par_to_z, "parity_from_z": par_from_z,
            "deg_inv_sqrt": deg_inv_sqrt,
        }

        scalar_feat = self.atom_embed(Z)
        x_data = scalar_feat.new_zeros(Z.shape[0], self.irreps_hidden.dim)
        if self._scalar_slice is not None:
            x_data[..., self._scalar_slice] = scalar_feat
        else:
            x_data[..., self._scalar_idx] = scalar_feat
        x = O3Tensor(x_data, self.irreps_hidden)

        for layer in self.layers:
            x = layer(x, edge_index, edge_geom=edge_geom,
                      edge_mask=edge_mask)

        x = self.out_norm(x)
        raw = x.tensor
        if self._scalar_slice is not None:
            atom_e = raw[..., self._scalar_slice]
        else:
            atom_e = raw[..., self._scalar_idx]  # (N, n_scalar)
        atom_e = self.energy_proj(atom_e).squeeze(-1)

        E_pred = torch.zeros(n_mol, device=atom_e.device, dtype=atom_e.dtype)
        E_pred = E_pred.scatter_add(0, mol_idx, atom_e)
        return atom_e.sum(), E_pred

    def forward(self, batch):
        """
        Parameters
        ----------
        batch : dict
            Z          : (total_atoms,)   atomic numbers
            R          : (total_atoms, 3) atom positions
            edge_index : (2, total_edges)
            batch      : (total_atoms,)   molecule index per atom

        Returns
        -------
        dict
            F : (total_atoms, 3)  predicted forces
            E : (batch_size,)     predicted energies
        """
        Z = batch["Z"]
        R = batch["R"]
        edge_index = batch["edge_index"]
        mol_idx = batch["batch"]
        n_mol = batch["N"].shape[0]
        edge_shift = batch.get("edge_shift", None)
        edge_mask = batch.get("edge_mask", None)

        # ---- Conservative forces: F = -dE/dR ------------------------------
        # One forward pass through the energy function, one autograd.grad back
        # to the positions.  At training time create_graph=True keeps the
        # second-order graph that loss.backward() needs; at eval it is dropped.
        energy_fn = self._compiled_energy if self._compiled_energy is not None \
            else self._energy
        # enable_grad so the force gradient is still taken when the caller runs
        # under torch.no_grad() (Trainer._evaluate does); eval outputs are
        # detached below so no graph escapes.
        with torch.enable_grad():
            R_g = R if R.requires_grad else R.detach().requires_grad_(True)
            total_E, E_pred = energy_fn(
                R_g, Z, edge_index, mol_idx, n_mol, edge_shift, edge_mask,
            )
            create_graph = self.training
            dE_dR, = torch.autograd.grad(
                outputs=total_E,
                inputs=R_g,
                create_graph=create_graph,
                retain_graph=create_graph,
            )
        F_pred = -dE_dR

        if not self.training:
            # Eval: drop the autograd graph so eval batches don't accumulate it.
            F_pred = F_pred.detach()
            E_pred = E_pred.detach()
        return {"F": F_pred, "E": E_pred}
