"""eSCN-style decoupled-frame construction of the edge Wigner-D caches.

Drop-in replacement for ``batched_rotation_to_z`` + two
``precompute_wigner_D_from_matrix`` calls in the K/V edge projection.

Why
---
The lab-z construction rotates the edge to +z, so R33 = d_z and the Euler gimbal
``beta = acos(R33)`` sits EXACTLY at 0/pi for every axis-aligned edge (systematic
in periodic slabs) -> catastrophic fp32 force noise.  eSCN builds the frame by
Gram-Schmidt from the edge + an auxiliary vector, aligning the edge to a NON-z
reference so R33 is a frame quantity DECOUPLED from d_z; axis-aligned edges are no
longer systematically at the gimbal.

Model function is UNCHANGED.  Two frames that both send the edge to +z differ by
an element of the stabiliser of +z, so ``D(R)^-1 M D(R)`` is frame-independent
whenever the block ``M`` commutes with that stabiliser.  ``escn_frame`` is
PROPER by construction (det = +1, and ``escn_wigner_D_caches`` returns
parity = +1 everywhere), so the gauge change here lives entirely in SO(2) -- no
reflection is involved.

* ``O2Linear``'s block is ``w (x) I_2``, which commutes with the FULL O(2)
  stabiliser (rotations and the reflection alike), so the swap preserves its
  O(3)-equivariant operation exactly (bit-close away from poles).
* ``SO2Linear``'s block ``[[A, -B], [B, A]]`` commutes with the SO(2) rotations
  but not with the reflection.  That is BY DESIGN: an SO(2) linear makes the
  model SO(3)-equivariant rather than O(3)-equivariant, which is what the eSCN
  construction is for -- not a hazard.  Because the eSCN gauge change is itself
  a proper rotation, ``SO2Linear`` is equally unaffected by the swap.

Construction
------------
* Proper frame ``R_escn`` (det = +1) mapping the unit edge -> +y, built from the
  edge and a fixed generic auxiliary (two candidates; pick the one less aligned
  with the edge so cross products stay well-conditioned).  Rows ``[nz, d, ny]``
  with ``nz = norm(d x aux)``, ``ny = norm(nz x d)``; then ``R d = +y`` and
  ``det = nz . (d x ny) = nz . nz = +1``.
* Compose a FIXED proper rotation ``Rfix`` (`` y -> z``) so the +z convention the
  O2Linear expects is preserved: ``D_to_z = D(Rfix) @ D(R_escn)`` (Wigner-D is a
  representation, so this equals ``D(Rfix @ R_escn)`` -- the Wigner-D of the
  edge->+z rotation) WITHOUT ever extracting Euler angles from the d_z-coupled
  matrix.  ``D_from_z = D_to_z^T`` (exact inverse).
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import Tensor

from .wigner import precompute_wigner_D_from_matrix

# Fixed generic unit auxiliaries: non-axis-aligned and well separated from each
# other, so at least one is far from any given edge direction.
_AUX1 = torch.tensor([0.31225, 0.59341, 0.74231])
_AUX1 = _AUX1 / _AUX1.norm()
_AUX2 = torch.tensor([0.74231, -0.31225, 0.59341])
_AUX2 = _AUX2 / _AUX2.norm()

# Fixed proper rotation mapping +y -> +z  (Rx(+90deg): y->z, z->-y).  det = +1.
_RFIX = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0]])

_dfix_cache: Dict[tuple, Dict[int, Tensor]] = {}

# Pole-free Euler parametrisation.  matrix_to_euler_ZYZ is singular where the
# frame's R_33 = +-1 (and snaps beta there, dropping its gradient).  R_33 of the
# eSCN frame is ny . e_z and CAN reach +-1.  Instead evaluate
# D(R) = D(R Q^T) D(Q) for a fixed proper rotation Q whose third row is
# q = aux1 x aux2: then (R Q^T)_33 = ny . q, and since ny . aux > 0.43 for the
# auxiliary in use, |ny . q| <= 0.9 for EVERY edge direction (beta in
# [26, 154] deg).  Exact identity of the representation, no new parameters.
_Q_ROW3 = torch.linalg.cross(_AUX1.double(), _AUX2.double())
_Q_ROW3 = _Q_ROW3 / _Q_ROW3.norm()
_Q_ROW1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
_Q_ROW1 = _Q_ROW1 - (_Q_ROW1 @ _Q_ROW3) * _Q_ROW3
_Q_ROW1 = _Q_ROW1 / _Q_ROW1.norm()
_QPOLE = torch.stack([_Q_ROW1, torch.linalg.cross(_Q_ROW3, _Q_ROW1), _Q_ROW3])   # det +1
_dq_cache: Dict[tuple, Dict[int, Tensor]] = {}


def _get_dq(ls, device, dtype) -> Dict[int, Tensor]:
    """Wigner-D of the fixed pole-avoiding rotation Q, per degree (cached, (2l+1,2l+1))."""
    key = (tuple(sorted(ls)), device, str(dtype))
    if key not in _dq_cache:
        cache, _ = precompute_wigner_D_from_matrix(ls, _QPOLE.unsqueeze(0))   # float64
        _dq_cache[key] = {l: v[0].to(device=device, dtype=dtype) for l, v in cache.items()}
    return _dq_cache[key]


def _get_dfix(ls, device, dtype) -> Dict[int, Tensor]:
    """Wigner-D of the fixed y->z rotation, per degree (cached, shape (1,2l+1,2l+1))."""
    key = (tuple(sorted(ls)), device, str(dtype))
    if key not in _dfix_cache:
        Rfix = _RFIX.to(device=device, dtype=dtype).unsqueeze(0)
        cache, _ = precompute_wigner_D_from_matrix(ls, Rfix)
        _dfix_cache[key] = cache
    return _dfix_cache[key]


def escn_frame(d: Tensor) -> Tensor:
    """Proper rotation ``(E,3,3)`` mapping unit ``d`` -> +y, decoupled from lab axes."""
    aux1 = _AUX1.to(device=d.device, dtype=d.dtype)
    aux2 = _AUX2.to(device=d.device, dtype=d.dtype)
    # Pick the auxiliary less aligned with the edge (keeps d x aux well away from 0).
    use2 = ((d * aux1).sum(-1, keepdim=True).abs() > 0.9)
    aux = torch.where(use2, aux2, aux1).expand_as(d)
    nz = torch.linalg.cross(d, aux)
    nz = nz / nz.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    ny = torch.linalg.cross(nz, d)
    ny = ny / ny.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.stack([nz, d, ny], dim=-2)   # rows -> R d = +y, det = +1


def escn_wigner_D_caches(
    directions: Tensor, ls: Iterable[int], pole_free: bool = True,
) -> Tuple[Dict[int, Tensor], Tensor, Dict[int, Tensor], Tensor]:
    """Pole-free ``(D_to_z, parity_to_z, D_from_z, parity_from_z)`` for edge dirs.

    ``directions`` : ``(E,3)`` (need not be unit).  Returns caches keyed by degree,
    matching ``precompute_wigner_D_from_matrix`` so this is a drop-in swap.  Parity
    is +1 everywhere (the frame is a proper rotation by construction).

    ``pole_free=True`` (the model's ``escn_frame="polefree"``) evaluates
    D(R) = D(R Q^T) D(Q), so the Euler angles never reach a pole and forces are
    exactly equivariant for every edge direction.  ``pole_free=False``
    (``escn_frame="euler"``) takes D(R) straight from the ZYZ Euler angles of R.  Its
    snapped pole branch
    carries no gradient through beta, so for edges within ~1.5e-3 rad of the gimbal
    directions (~2e-6 of all directions) the frame's force contribution is dropped.
    Both give the same D up to float round-off.
    """
    ls = list(ls)
    d = directions.reshape(-1, 3)
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    R_escn = escn_frame(d)
    if pole_free:
        # R_escn @ Q^T and D(R Q^T) @ D(Q) as broadcast-multiply-and-sum, NOT matmul,
        # so TF32 ("high" matmul precision) never touches the frame or D.
        Q = _QPOLE.to(device=d.device, dtype=d.dtype)
        R_off = (R_escn.unsqueeze(-2) * Q).sum(-1)
        D_off, _ = precompute_wigner_D_from_matrix(ls, R_off)
        DQ = _get_dq(ls, d.device, d.dtype)
        D_escn = {l: (D_off[l].unsqueeze(-1) * DQ[l]).sum(-2) for l in D_off}
    else:
        D_escn, _ = precompute_wigner_D_from_matrix(ls, R_escn)
    Dfix = _get_dfix(ls, d.device, d.dtype)
    D_to: Dict[int, Tensor] = {l: Dfix[l] @ D_escn[l] for l in D_escn}
    D_from: Dict[int, Tensor] = {l: D_to[l].transpose(-1, -2) for l in D_to}
    par = d.new_ones(d.shape[0])
    return D_to, par, D_from, par
