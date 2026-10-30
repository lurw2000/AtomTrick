"""Wigner-D matrices and irrep rotation (CPU, float64).

``wigner_D`` routes through the cached ``WignerDSmall`` evaluator; it is checked
here against the textbook construction (real <- complex change of basis around
the small-d matrix, ZYZ phases), in value and in gradient -- the model's forces
differentiate through it.  The Euler extraction is checked at the gimbal poles
(beta = 0, pi), where commit 0c46084 restored beta exactly (before it, D was off
by ~1e-3 for every axis-aligned rotation).

Replaces the GPU scripts ``test_wigner_grad_direct.py``,
``test_wigner_small_equiv.py``, ``test_wigner_small_fixed.py``,
``test_wigner_fp32_precision.py`` and ``test_rotate_equiv.py``.
"""
import math

import pytest
import torch

from AtomTrick.o3.irreps import Irreps
from AtomTrick.o3.wigner import (_change_of_basis_real_complex, euler_ZYZ_to_matrix,
                                 matrix_to_euler_ZYZ, precompute_wigner_D_from_matrix,
                                 rotate_irreps, wigner_D, wigner_D_from_matrix, wigner_d)

from common import improper_elements, random_rotation

DT = torch.float64
LMAX = 4


def reference_wigner_D(l, alpha, beta, gamma):
    """D = U diag(e^{-i m alpha}) d(beta) diag(e^{-i m gamma}) U^H (real basis)."""
    if l == 0:
        return torch.ones(*alpha.shape, 1, 1, dtype=alpha.dtype)
    d = wigner_d(l, beta).to(torch.complex128)
    ms = torch.arange(-l, l + 1, dtype=alpha.dtype)
    U = _change_of_basis_real_complex(l, dtype=alpha.dtype, device=alpha.device)
    pa = torch.exp(-1j * ms * alpha[..., None])
    pg = torch.exp(-1j * ms * gamma[..., None])
    Dc = pa[..., :, None] * d * pg[..., None, :]
    return (U @ Dc @ U.conj().T).real.to(alpha.dtype)


def _angles(n, seed, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, generator=g, dtype=DT) * 2 * math.pi
    b = torch.rand(n, generator=g, dtype=DT) * math.pi
    c = torch.rand(n, generator=g, dtype=DT) * 2 * math.pi
    return [t.requires_grad_(requires_grad) for t in (a, b, c)]


@pytest.mark.parametrize("l", range(1, LMAX + 1))
def test_wigner_D_matches_reference_value_and_gradient(l):
    a, b, c = _angles(64, seed=l, requires_grad=True)
    sens = torch.randn(64, 2 * l + 1, 2 * l + 1, dtype=DT)
    D = wigner_D(l, a, b, c)
    grads = torch.autograd.grad((D * sens).sum(), (a, b, c))
    a2, b2, c2 = [t.detach().clone().requires_grad_(True) for t in (a, b, c)]
    D_ref = reference_wigner_D(l, a2, b2, c2)
    grads_ref = torch.autograd.grad((D_ref * sens).sum(), (a2, b2, c2))
    assert float((D - D_ref).abs().max()) < 1e-12
    for g, g_ref in zip(grads, grads_ref):
        assert float((g - g_ref).abs().max()) < 1e-10 * max(1.0, float(g_ref.abs().max()))


@pytest.mark.parametrize("l", range(1, LMAX + 1))
def test_wigner_D_is_orthogonal(l):
    a, b, c = _angles(64, seed=10 + l)
    for dtype, tol in ((DT, 1e-12), (torch.float32, 1e-5)):
        D = wigner_D(l, a.to(dtype), b.to(dtype), c.to(dtype))
        eye = torch.eye(2 * l + 1, dtype=dtype)
        assert float((D @ D.transpose(-1, -2) - eye).abs().max()) < tol


def _rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=DT)


def _rx(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=DT)


#: rotations whose R33 is exactly +-1 (beta = 0 or pi): the Euler gimbal
GIMBAL = {
    "identity": torch.eye(3, dtype=DT),
    "rz": _rz(0.7),
    "rx_pi": _rx(math.pi),
    "rx_pi_rz": _rx(math.pi) @ _rz(1.9),
}


@pytest.mark.parametrize("name", sorted(GIMBAL))
def test_euler_roundtrip_at_the_gimbal(name):
    """At beta = pi only alpha - gamma is defined; with gamma = 0 the matrix is
    Rz(alpha) Ry(pi), whose (1,0) entry is -sin(alpha): alpha must be
    atan2(-R10, -R00).  (The model never reaches beta = pi -- the eSCN frame has
    R33 > -1 for every edge -- but rotate_irreps / O3Tensor.transform do for
    e.g. a rotation by pi about a horizontal axis.)"""
    R = GIMBAL[name]
    a, b, c = matrix_to_euler_ZYZ(R)
    err = float((euler_ZYZ_to_matrix(a, b, c) - R).abs().max())
    assert err < 1e-12, (f"Euler round trip off by {err:.2e} at {name}; if beta = pi, check "
                         "the degen_pi branch of matrix_to_euler_ZYZ (alpha = atan2(-R10, -R00))")


def test_euler_roundtrip_generic():
    Rs = torch.stack([random_rotation(s) for s in range(20)])
    a, b, c = matrix_to_euler_ZYZ(Rs)
    assert float((euler_ZYZ_to_matrix(a, b, c) - Rs).abs().max()) < 1e-12


@pytest.mark.parametrize("name", sorted(GIMBAL))
@pytest.mark.parametrize("l", [1, 2, 3])
def test_representation_law_including_the_gimbal(name, l):
    """D(R1 R2) = D(R1) D(R2), with R1 or R1 R2 on the gimbal."""
    R1, R2 = GIMBAL[name], random_rotation(l)
    D = lambda R: wigner_D_from_matrix(l, R)  # noqa: E731
    assert float((D(R1 @ R2) - D(R1) @ D(R2)).abs().max()) < 1e-10
    # the square of a z-rotation is the z-rotation by twice the angle
    assert float((D(_rz(1.4)) - D(_rz(0.7)) @ D(_rz(0.7))).abs().max()) < 1e-12


def _reference_rotate(irreps, R, x):
    """Block-by-block reference: D^l on each (mult, l) block, parity on odd irreps."""
    parity = torch.linalg.det(R).sign()
    R_proper = parity[..., None, None] * R
    a, b, c = matrix_to_euler_ZYZ(R_proper)
    parts = []
    for (mult, ir), sl in zip(irreps, irreps.slices()):
        D = reference_wigner_D(ir.l, a, b, c)
        if ir.p == -1:
            D = D * parity[..., None, None]
        blk = x[..., sl].unflatten(-1, (mult, ir.dim))
        parts.append(torch.einsum("...ij,...mj->...mi", D, blk).flatten(-2))
    return torch.cat(parts, dim=-1)


@pytest.mark.parametrize("improper", [False, True])
def test_rotate_irreps_matches_reference(improper):
    irreps = Irreps("3x0e+2x1o+2x2e+3x3o+1x1e")
    R = torch.stack([random_rotation(s) for s in range(8)])
    if improper:
        R = -R
    x = torch.randn(8, irreps.dim, dtype=DT)
    ref = _reference_rotate(irreps, R, x)
    assert float((rotate_irreps(irreps, R, x) - ref).abs().max()) < 1e-12
    D_cache, par = precompute_wigner_D_from_matrix(sorted({ir.l for _, ir in irreps}), R)
    out = rotate_irreps(irreps, None, x, parity=par, D_cache=D_cache)
    assert float((out - ref).abs().max()) < 1e-12


def test_rotate_irreps_is_a_group_action():
    irreps = Irreps("2x0e+2x1o+2x2e+2x3o")
    g1, g2 = random_rotation(1), improper_elements()["mirror_z_then_rotation"]
    x = torch.randn(1, irreps.dim, dtype=DT)
    lhs = rotate_irreps(irreps, (g1 @ g2).unsqueeze(0), x)
    rhs = rotate_irreps(irreps, g1.unsqueeze(0), rotate_irreps(irreps, g2.unsqueeze(0), x))
    assert float((lhs - rhs).abs().max()) < 1e-12
