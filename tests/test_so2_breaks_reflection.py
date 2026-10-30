"""O(2)- vs SO(2)-linear edge maps (the paper's O(2)-vs-SO(2) ablation).

Per |m| > 0 block, an SO(2)-equivariant linear map may use a complex-linear
block [[W_a, -W_b], [W_b, W_a]]; O(2) (which adds the reflection through a
plane containing the edge) forces W_b = 0 (paper App. B).  Consequences tested
here, at the level of a single edge map and of the full model:

* both maps are SO(3)-equivariant after conjugation with the edge frame;
* the O(2) map is also equivariant under reflections, the SO(2) map is NOT once
  W_b != 0.  ``SO2Linear`` zero-initialises W_b (the SO(2) control starts from
  the O(2) solution), so every test that expects a violation perturbs W_b first;
* a freshly built SO2Linear equals an O2Linear carrying its W_a;
* both maps stay finite and SO(3)-equivariant for edges exactly at the poles.

Folds in the former ``diag_so2_equiv.py`` and ``diag_so2_pole.py``.
"""
import math

import pytest
import torch

from AtomTrick.o3.escn_rotation import escn_wigner_D_caches
from AtomTrick.o3.irreps import Irreps
from AtomTrick.o3.linear import O2Linear, SO2Linear
from AtomTrick.o3.tensor import O3Tensor
from AtomTrick.o3.wigner import precompute_wigner_D_from_matrix, rotate_irreps

from common import build_model, improper_elements, make_batch, random_rotation, symmetry_errors

DT = torch.float64
IRREPS = Irreps("4x0e+4x1o+4x2e+4x3o")
LS = sorted({ir.l for _, ir in IRREPS})
N_EDGES = 7
TOL = 1e-10
BREAK = 1e-4          # a genuine violation is O(1e-1); this is a safe floor


def perturb_antisymmetric(module: torch.nn.Module, seed: int = 0) -> int:
    """Give every SO2Linear a non-zero W_b (the second slice of each stacked
    ``(2, mo, mi)`` |m|>0 block).  Returns the number of blocks touched."""
    g = torch.Generator().manual_seed(seed)
    touched = 0
    for m in module.modules():
        if isinstance(m, SO2Linear) and m.weight_blocks is not None:
            for blk in m.weight_blocks:
                if blk.dim() == 3:
                    std = 1.0 / math.sqrt(blk.shape[-1])
                    with torch.no_grad():
                        blk[1].copy_(torch.randn(blk[1].shape, generator=g,
                                                 dtype=torch.float64).to(blk.dtype) * std)
                    touched += 1
    assert touched > 0, "no SO2Linear |m|>0 block found"
    return touched


def edge_apply(lin, x, d):
    """The exact K/V path of the attention block: rotate into the (eSCN) edge
    frame, apply the per-|m| block map, rotate back."""
    D_to, p_to, D_from, p_from = escn_wigner_D_caches(d, LS)
    x_z = rotate_irreps(IRREPS, None, x, D_cache=D_to, parity=p_to)
    y_z = lin(O3Tensor(x_z, IRREPS)).tensor
    return rotate_irreps(IRREPS, None, y_z, D_cache=D_from, parity=p_from)


def edge_equivariance_err(lin, g, d=None, seed=0):
    """max |f(D(g)x, g d) - D(g) f(x, d)| over a batch of random edges."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(N_EDGES if d is None else d.shape[0], IRREPS.dim, generator=gen, dtype=DT)
    if d is None:
        d = torch.randn(N_EDGES, 3, generator=gen, dtype=DT)
    gg = g.to(DT).expand(d.shape[0], 3, 3).contiguous()
    Dg, pg = precompute_wigner_D_from_matrix(LS, gg)
    lhs = edge_apply(lin, rotate_irreps(IRREPS, None, x, D_cache=Dg, parity=pg), d @ g.to(DT).T)
    rhs = rotate_irreps(IRREPS, None, edge_apply(lin, x, d), D_cache=Dg, parity=pg)
    assert torch.isfinite(lhs).all() and torch.isfinite(rhs).all()
    return float((lhs - rhs).abs().max())


@pytest.fixture
def o2():
    torch.manual_seed(0)
    return O2Linear(IRREPS, IRREPS, internal=True, bias=False).to(DT)


@pytest.fixture
def so2():
    torch.manual_seed(1)
    lin = SO2Linear(IRREPS, IRREPS, internal=True, bias=False).to(DT)
    perturb_antisymmetric(lin)
    return lin


# ---------------------------------------------------------------------------
# Single edge map
# ---------------------------------------------------------------------------

def test_so2_has_more_parameters(o2, so2):
    """SO(2) adds one extra C x C matrix per (|m|>0) block: exactly 2x on |m|>0."""
    blocks_o2 = [b.numel() for b in o2.weight_blocks]
    blocks_so2 = [b.numel() for b in so2.weight_blocks]
    assert blocks_so2[0] == blocks_o2[0]                       # |m| = 0 identical
    assert blocks_so2[1:] == [2 * n for n in blocks_o2[1:]]


@pytest.mark.parametrize("seed", [0, 1])
def test_both_are_rotation_equivariant(o2, so2, seed):
    g = random_rotation(seed)
    assert edge_equivariance_err(o2, g) < TOL
    assert edge_equivariance_err(so2, g) < TOL


@pytest.mark.parametrize("name", ["inversion", "mirror_x", "mirror_z_then_rotation"])
def test_only_o2_is_reflection_equivariant(o2, so2, name):
    g = improper_elements()[name]
    assert edge_equivariance_err(o2, g) < TOL, "O2Linear must be O(3)-equivariant"
    err = edge_equivariance_err(so2, g)
    assert err > BREAK, (f"SO2Linear with W_b != 0 should break {name} "
                         f"(err {err:.2e}); is W_b being used?")


@pytest.mark.parametrize("theta", [0.3, 1.1, 2.7])
def test_blocks_commute_with_rotations_about_the_axis(o2, so2, theta):
    """Gauge invariance: the block map commutes with D(R_z(theta)), so the
    arbitrary choice of frame around the edge cannot leak into the output."""
    c, s = math.cos(theta), math.sin(theta)
    Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=DT)
    Dz, pz = precompute_wigner_D_from_matrix(LS, Rz.expand(N_EDGES, 3, 3).contiguous())
    x = torch.randn(N_EDGES, IRREPS.dim, dtype=DT)
    for lin in (o2, so2):
        lhs = lin(O3Tensor(rotate_irreps(IRREPS, None, x, D_cache=Dz, parity=pz), IRREPS)).tensor
        rhs = rotate_irreps(IRREPS, None, lin(O3Tensor(x, IRREPS)).tensor, D_cache=Dz, parity=pz)
        assert float((lhs - rhs).abs().max()) < TOL, type(lin).__name__


def test_only_the_o2_block_commutes_with_the_axis_reflection(o2, so2):
    """Reflection through the xz-plane (which contains the edge axis z): together
    with R_z(theta) it generates O(2).  W (x) I_2 commutes with it; the SO(2)
    block's antisymmetric W_b part anticommutes (paper App. B)."""
    g = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=DT))
    Dg, pg = precompute_wigner_D_from_matrix(LS, g.expand(N_EDGES, 3, 3).contiguous())
    x = torch.randn(N_EDGES, IRREPS.dim, dtype=DT)
    errs = {}
    for name, lin in (("o2", o2), ("so2", so2)):
        lhs = lin(O3Tensor(rotate_irreps(IRREPS, None, x, D_cache=Dg, parity=pg), IRREPS)).tensor
        rhs = rotate_irreps(IRREPS, None, lin(O3Tensor(x, IRREPS)).tensor, D_cache=Dg, parity=pg)
        errs[name] = float((lhs - rhs).abs().max())
    assert errs["o2"] < TOL and errs["so2"] > BREAK, errs


def test_fresh_so2_equals_o2_with_same_wa():
    """W_b is zero-initialised: an untouched SO2Linear IS an O2Linear."""
    torch.manual_seed(2)
    so2 = SO2Linear(IRREPS, IRREPS, internal=True, bias=False).to(DT)
    o2 = O2Linear(IRREPS, IRREPS, internal=True, bias=False).to(DT)
    with torch.no_grad():
        for b_o2, b_so2 in zip(o2.weight_blocks, so2.weight_blocks):
            if b_so2.dim() == 3:
                assert torch.count_nonzero(b_so2[1]) == 0, "W_b must be zero-initialised"
                b_o2.copy_(b_so2[0])
            else:
                b_o2.copy_(b_so2)
    x = torch.randn(N_EDGES, IRREPS.dim, dtype=DT)
    d = torch.randn(N_EDGES, 3, dtype=DT)
    assert float((edge_apply(so2, x, d) - edge_apply(o2, x, d)).abs().max()) < TOL


def test_pole_edges_finite_and_equivariant(o2, so2):
    """Edges exactly (and almost) along +-z, the old lab-frame gimbal; the model's
    eSCN frame must keep both maps finite and SO(3)-equivariant there."""
    d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                      [1e-7, 0.0, 1.0], [0.0, 1e-7, -1.0]], dtype=DT)
    for lin in (o2, so2):
        for seed in (3, 4):
            assert edge_equivariance_err(lin, random_rotation(seed), d=d, seed=seed) < 1e-9


def test_so2_output_is_smooth_through_the_pole():
    """fp32 sweep of the edge direction across +z: no spike at the pole
    (a gimbal-locking frame shows up as a jump there, i.e. a force spike)."""
    torch.manual_seed(5)
    lin = SO2Linear(IRREPS, IRREPS, internal=True, bias=False)
    perturb_antisymmetric(lin, seed=5)
    th = torch.linspace(-0.02, 0.02, 161)
    d = torch.stack([torch.sin(th), torch.zeros_like(th), torch.cos(th)], -1)
    x = torch.randn(1, IRREPS.dim).expand(161, -1).contiguous()
    D_to, p_to, D_from, p_from = escn_wigner_D_caches(d, LS)
    y = rotate_irreps(IRREPS, None,
                      lin(O3Tensor(rotate_irreps(IRREPS, None, x, D_cache=D_to, parity=p_to),
                                   IRREPS)).tensor,
                      D_cache=D_from, parity=p_from)
    assert torch.isfinite(y).all()
    jumps = (y[1:] - y[:-1]).abs().amax(-1)
    # The step is 2.5e-4 rad; a smooth map moves by ~|y| * step per step.  The
    # jump across the pole must look like every other jump.
    assert jumps[79:81].max() < 5 * jumps.median() + 1e-5


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def so2_model():
    model = build_model(dtype=torch.float64, seed=6, kv="so2", n_layers=3)
    perturb_antisymmetric(model, seed=6)
    return model


@pytest.fixture(scope="module")
def batch():
    return make_batch((6, 5), dtype=torch.float64, seed=0)


@pytest.mark.parametrize("seed", [0, 1])
def test_so2_model_is_rotation_equivariant(so2_model, batch, seed):
    e_err, f_err = symmetry_errors(so2_model, batch, random_rotation(seed))
    assert e_err < 1e-8 and f_err < 1e-8, (e_err, f_err)


@pytest.mark.parametrize("name", ["inversion", "mirror_x"])
def test_so2_model_breaks_reflection(so2_model, batch, name):
    e_err, f_err = symmetry_errors(so2_model, batch, improper_elements()[name])
    assert max(e_err, f_err) > BREAK, (
        f"SO(2) model looks {name}-equivariant (E {e_err:.2e}, F {f_err:.2e})")


def test_so2_model_at_initialisation_is_still_o3_equivariant(batch):
    """With W_b = 0 (fresh init) the SO(2) model is the O(2) model, so the
    reflection break above is caused by W_b and nothing else."""
    model = build_model(dtype=torch.float64, seed=6, kv="so2", n_layers=3)
    e_err, f_err = symmetry_errors(model, batch, improper_elements()["inversion"])
    assert e_err < 1e-8 and f_err < 1e-8, (e_err, f_err)
