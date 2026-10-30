"""Unit tests of the building blocks (CPU, float64).

* ``O3Linear`` is O(3)-equivariant and equals its dense block-diagonal matrix;
* ``O2Linear`` equals its dense per-|m| matrix (the node-frame path);
* the gated activation and both LayerNorms are O(3)-equivariant; the shared
  LayerNorm centres the scalars and divides by ONE RMS over all degrees;
* the radial basis and hence every edge message vanish smoothly at the cutoff,
  so the energy is continuous (in fact C^3) as atoms cross it -- a property of
  identity attention with the bias-free radial gate MLP.

Replaces the GPU benchmark scripts ``test_o3linear_equiv.py``,
``test_o2linear_equiv.py``, ``test_activation_equiv.py`` and the unit part of
``test_ffn_act_ablation.py`` (timing code dropped).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from AtomTrick.o3.activation import O3GatedActivation
from AtomTrick.o3.irreps import Irreps
from AtomTrick.o3.linear import O2Linear, O3Linear
from AtomTrick.o3.norm import O3LayerNorm, O3SharedLayerNorm
from AtomTrick.o3.tensor import O3Tensor
from AtomTrick.o3.wigner import rotate_irreps
from AtomTrick.radial import ExpNormalSmearing

from common import batch_from, build_model, improper_elements, predict, random_rotation

DT = torch.float64
IRREPS = Irreps("4x0e+4x1o+4x2e+4x3o")
TOL = 1e-11


def _rot(irreps, g, x):
    return rotate_irreps(irreps, g.to(DT).expand(x.shape[0], 3, 3).contiguous(), x)


GROUP = {"rotation": random_rotation(0), **improper_elements()}


# ---------------------------------------------------------------------------
# Linear maps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("g_name", sorted(GROUP))
def test_o3linear_is_o3_equivariant(g_name):
    torch.manual_seed(0)
    irreps_out = Irreps("6x0e+2x1o+3x2e+5x3o")
    lin = O3Linear(IRREPS, irreps_out, bias=True).to(DT)
    with torch.no_grad():
        lin.bias.normal_()
    x = torch.randn(5, IRREPS.dim, dtype=DT)
    g = GROUP[g_name]
    lhs = lin(O3Tensor(_rot(IRREPS, g, x), IRREPS)).tensor
    rhs = _rot(irreps_out, g, lin(O3Tensor(x, IRREPS)).tensor)
    assert float((lhs - rhs).abs().max()) < TOL


def _require(obj, *names):
    missing = [n for n in names if not hasattr(obj, n)]
    if missing:
        pytest.skip(f"internal attributes changed: {missing}")


def test_o3linear_matches_dense_reference():
    torch.manual_seed(1)
    lin = O3Linear(IRREPS, IRREPS, bias=True).to(DT)
    with torch.no_grad():
        lin.bias.normal_()
    _require(lin, "_row_idx", "_col_idx", "_expand_idx", "_in_gather", "_out_scatter",
             "_bias_slice", "_in_dim", "_out_dim")

    def dense(raw):
        if lin._in_gather is not None:
            raw = raw[..., lin._in_gather]
        W = raw.new_zeros(lin._out_dim, lin._in_dim)
        W[lin._row_idx, lin._col_idx] = lin.weight[lin._expand_idx]
        out = F.linear(raw, W)
        if lin.bias is not None and lin._bias_slice is not None:
            out[..., lin._bias_slice] = out[..., lin._bias_slice] + lin.bias
        if lin._out_scatter is not None:
            out = out[..., lin._out_scatter]
        return out

    x = torch.randn(9, IRREPS.dim, dtype=DT, requires_grad=True)
    y = lin(O3Tensor(x, IRREPS)).tensor
    y_ref = dense(x)
    assert float((y - y_ref).abs().max()) < TOL
    g = torch.randn_like(y)
    gx, = torch.autograd.grad(y, x, g)
    gx_ref, = torch.autograd.grad(y_ref, x, g)
    assert float((gx - gx_ref).abs().max()) < TOL


def test_o2linear_matches_dense_reference():
    torch.manual_seed(2)
    lin = O2Linear(IRREPS, IRREPS, internal=True, bias=False).to(DT)
    _require(lin, "_row_idx", "_col_idx", "_expand_idx", "_in_dim", "_out_dim")
    x = torch.randn(9, IRREPS.dim, dtype=DT, requires_grad=True)
    W = x.new_zeros(lin._out_dim, lin._in_dim)
    W[lin._row_idx, lin._col_idx] = lin.weight[lin._expand_idx]
    y = lin(O3Tensor(x, IRREPS)).tensor
    y_ref = F.linear(x, W)
    assert float((y - y_ref).abs().max()) < TOL
    g = torch.randn_like(y)
    assert float((torch.autograd.grad(y, x, g)[0]
                  - torch.autograd.grad(y_ref, x, g)[0]).abs().max()) < TOL


def test_o2linear_block_sizes():
    """One dense matrix per |m| over every (l, c) with l >= |m| (paper Eq. 8)."""
    C = 4
    lin = O2Linear(IRREPS, IRREPS, internal=True, bias=False)
    sizes = [tuple(b.shape) for b in lin.weight_blocks]
    assert sizes == [((4 - m) * C, (4 - m) * C) for m in range(4)]


# ---------------------------------------------------------------------------
# Activation and normalisation
# ---------------------------------------------------------------------------

def test_gated_activation_layout():
    act = O3GatedActivation(Irreps("8x0e+8x1o+4x2e"))
    assert act._n_scalar_features == 8
    assert act._n_gate_scalars == 12                   # one gate per 1o/2e channel
    assert act.irreps_in.dim == 8 + 12 + 8 * 3 + 4 * 5


@pytest.mark.parametrize("g_name", ["rotation", "inversion", "mirror_x"])
def test_gated_activation_is_equivariant(g_name):
    act = O3GatedActivation(Irreps("8x0e+8x1o+4x2e+4x3o"))
    x = torch.randn(6, act.irreps_in.dim, dtype=DT)
    g = GROUP[g_name]
    lhs = act(O3Tensor(_rot(act.irreps_in, g, x), act.irreps_in)).tensor
    rhs = _rot(act.irreps_out, g, act(O3Tensor(x, act.irreps_in)).tensor)
    assert float((lhs - rhs).abs().max()) < TOL


@pytest.mark.parametrize("norm_cls", [O3SharedLayerNorm, O3LayerNorm])
@pytest.mark.parametrize("g_name", ["rotation", "rotoinversion"])
def test_layernorms_are_equivariant(norm_cls, g_name):
    torch.manual_seed(3)
    ln = norm_cls(IRREPS).to(DT)
    with torch.no_grad():
        for p in ln.parameters():
            p.normal_()
    x = torch.randn(6, IRREPS.dim, dtype=DT)
    g = GROUP[g_name]
    lhs = ln(O3Tensor(_rot(IRREPS, g, x), IRREPS)).tensor
    rhs = _rot(IRREPS, g, ln(O3Tensor(x, IRREPS)).tensor)
    assert float((lhs - rhs).abs().max()) < TOL


def test_shared_layernorm_normalises_all_degrees_jointly():
    """At init (unit gain, zero bias): scalars are centred, and the mean square
    over ALL components of a node is 1 -- one RMS for every degree."""
    ln = O3SharedLayerNorm(IRREPS).to(DT)
    x = torch.randn(6, IRREPS.dim, dtype=DT) * torch.linspace(0.5, 3.0, IRREPS.dim, dtype=DT)
    y = ln(O3Tensor(x, IRREPS)).tensor
    scalars = y[:, :4]                                    # the leading 4x0e block
    assert float(scalars.mean(-1).abs().max()) < 1e-12
    ms = y.pow(2).mean(-1)
    assert float((ms - 1).abs().max()) < 1e-3             # eps = 1e-5 in the denominator
    # per-degree mean squares are NOT individually 1 (unlike per-degree LN)
    per_l = [y[:, 4:16].pow(2).mean(-1), y[:, 36:64].pow(2).mean(-1)]
    assert float((per_l[0] - per_l[1]).abs().max()) > 1e-2


# ---------------------------------------------------------------------------
# Radial basis and smoothness at the cutoff
# ---------------------------------------------------------------------------

def test_radial_basis_vanishes_smoothly_at_cutoff():
    rbf = ExpNormalSmearing(8, 5.0).to(DT)
    d = torch.tensor([4.999, 5.0, 5.2], dtype=DT, requires_grad=True)
    phi = rbf(d)
    assert float(phi[1:].abs().max()) == 0.0
    grad, = torch.autograd.grad(phi.sum(), d)
    assert float(grad[1:].abs().max()) == 0.0
    assert float(phi[0].abs().max()) < 1e-5 and float(grad[0].abs().max()) < 1e-2


def _three_atoms(r_ac):
    """A at the origin, B fixed, C placed at distance r_ac from A (|BC| < 5)."""
    x = math.sqrt(r_ac ** 2 - 0.64)
    R = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [x, 0.8, 0.0]], dtype=DT)
    return batch_from(torch.tensor([6, 8, 1]), R, torch.zeros(3, dtype=torch.long))


def _drop_edge(batch, i, j):
    src, dst = batch["edge_index"]
    keep = ~(((src == i) & (dst == j)) | ((src == j) & (dst == i)))
    out = dict(batch)
    out["edge_index"] = batch["edge_index"][:, keep]
    return out


def test_energy_is_continuous_across_the_cutoff():
    model = build_model(dtype=DT, seed=9)
    # Well inside the cutoff the A-C edge matters (the test is not vacuous) ...
    near = _three_atoms(4.0)
    E_with, _ = predict(model, near)
    E_without, _ = predict(model, _drop_edge(near, 0, 2))
    assert float((E_with - E_without).abs()) > 1e-6
    # ... but just inside the cutoff it contributes nothing measurable: messages
    # scale like (r_c - d)^4, so the edge can appear/disappear without a jump.
    edge = _three_atoms(5.0 - 1e-3)
    assert edge["edge_index"].shape[1] == 6
    E_in, F_in = predict(model, edge)
    E_out, F_out = predict(model, _drop_edge(edge, 0, 2))
    assert float((E_in - E_out).abs()) < 1e-9
    assert float((F_in - F_out).abs().max()) < 1e-6 * float(F_in.abs().max())
    # ... and across the boundary itself (the graph loses the edge at d >= 5):
    # the energy change equals the work of the forces (trapezoid rule, error
    # O(step^3)), i.e. there is no jump when the edge disappears.
    a, b = _three_atoms(5.0 - 1e-4), _three_atoms(5.0 + 1e-4)
    E_a, F_a = predict(model, a)
    E_b, F_b = predict(model, b)
    work = -0.5 * float(((F_a + F_b) * (b["R"] - a["R"])).sum())
    assert abs(float(E_b - E_a) - work) < 1e-9
