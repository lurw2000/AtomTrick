"""Symmetries of the full model: E(3) invariance of the energy, equivariance of
the forces, and permutation invariance.

The model under test is the final paper configuration (O(2)-linear K/V with
radial gates, identity attention, shared LayerNorm, gated FFN, scalar readout,
Lmax 3) at reduced width/depth, in float64 so that any violation above ~1e-10
is a real bug rather than round-off.  Absorbs the former
``benchmarks/equivariance_minimal.py``, ``tests/test_model.py``,
``tests/test_kv_ablation.py``, ``tests/test_ffn_act_ablation.py`` and the pole
checks of ``diag_so2_pole.py``.

Why improper elements matter: the O(2)-linear edge map mixes degrees l != l'
inside each |m| block.  That is O(3)-equivariant only because every irrep has
natural parity p = (-1)^l (0e, 1o, 2e, 3o) -- so reflections and inversion are
genuinely tested here, not implied by the rotation tests.
"""
import pytest
import torch

from common import (build_model, improper_elements, make_batch, max_rel_err, predict,
                    random_rotation, symmetry_errors, with_positions, batch_from)

TOL = 1e-8           # fp64: observed violations are ~1e-13 .. 1e-11
POLE_TOL = 1e-6      # edges exactly at an Euler gimbal (see test_gimbal_edges)


def test_forces_are_not_trivial(small_model, small_batch):
    """Guard against vacuous symmetry tests (zero or non-finite outputs)."""
    E, F = predict(small_model, small_batch)
    assert torch.isfinite(E).all() and torch.isfinite(F).all()
    assert F.abs().max() > 1e-4, "forces are ~0: symmetry tests would be vacuous"
    assert E.shape == (2,) and F.shape == small_batch["R"].shape


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotation_equivariance(small_model, small_batch, seed):
    e_err, f_err = symmetry_errors(small_model, small_batch, random_rotation(seed))
    assert e_err < TOL, f"energy not rotation-invariant: rel err {e_err:.2e}"
    assert f_err < TOL, f"forces not rotation-equivariant: rel err {f_err:.2e}"


@pytest.mark.parametrize("name", sorted(improper_elements()))
def test_improper_equivariance(small_model, small_batch, name):
    """Reflections / inversion: the O(2) model is O(3)-, not just SO(3)-equivariant."""
    g = improper_elements()[name]
    assert torch.linalg.det(g) < 0
    e_err, f_err = symmetry_errors(small_model, small_batch, g)
    assert e_err < TOL, f"energy not invariant under {name}: rel err {e_err:.2e}"
    assert f_err < TOL, f"forces not equivariant under {name}: rel err {f_err:.2e}"


def test_translation_invariance(small_model, small_batch):
    E, F = predict(small_model, small_batch)
    shift = torch.tensor([3.7, -12.1, 0.45], dtype=torch.float64)
    E2, F2 = predict(small_model, with_positions(small_batch, small_batch["R"] + shift))
    assert max_rel_err(E2, E) < TOL
    assert max_rel_err(F2, F) < TOL


def test_net_force_vanishes(small_model, small_batch):
    """Translation invariance of E  =>  sum_i F_i = 0 for every molecule."""
    _, F = predict(small_model, small_batch)
    net = torch.zeros(2, 3, dtype=F.dtype).index_add_(0, small_batch["batch"], F)
    assert net.abs().max() < 1e-10 * F.abs().max()


def test_permutation_invariance(small_model, small_batch):
    """Relabel atoms inside each molecule; the graph is rebuilt from scratch, so
    the edge order changes too."""
    E, F = predict(small_model, small_batch)
    g = torch.Generator().manual_seed(7)
    perm, off = [], 0
    for n in small_batch["N"].tolist():
        perm.append(torch.randperm(n, generator=g) + off)
        off += n
    perm = torch.cat(perm)
    permuted = batch_from(small_batch["Z"][perm], small_batch["R"][perm],
                          small_batch["batch"][perm])
    E2, F2 = predict(small_model, permuted)
    assert max_rel_err(E2, E) < TOL
    assert max_rel_err(F2, F[perm]) < TOL


def test_molecules_in_a_batch_do_not_interact(small_model, small_batch):
    """Batching is block-diagonal: a molecule's E/F do not depend on its batch-mates."""
    E, F = predict(small_model, small_batch)
    first = small_batch["batch"] == 0
    alone = batch_from(small_batch["Z"][first], small_batch["R"][first],
                       small_batch["batch"][first])
    E1, F1 = predict(small_model, alone)
    assert max_rel_err(E1, E[:1]) < TOL
    assert max_rel_err(F1, F[first]) < TOL


def _gimbal_molecule():
    """Atoms placed so that some edges sit EXACTLY on an Euler-angle gimbal.

    * +-z edges: the gimbal of the old lab-frame rotate-to-z construction.
    * edges along the xy-projection of the eSCN auxiliary vector: there the
      eSCN frame's third row is +-z, i.e. R33 = +-1 -- the gimbal of the frame
      the model actually uses (fixed in commit 0c46084, which restores beta
      exactly at the poles; before it, beta sat at 4.5e-4 and D was off by 1e-3).
      That fix makes the D VALUES exact, but the snapped branch carries no
      gradient through beta, so escn_frame="euler" forces are still incomplete
      there (test_gimbal_edges_forces_euler_frame).  The default pole-free
      evaluation never reaches the snapped branch (test_gimbal_edges_forces).
    """
    try:
        from AtomTrick.o3.escn_rotation import _AUX1
        aux = _AUX1.to(torch.float64)
    except ImportError:  # pragma: no cover - private symbol renamed
        aux = torch.tensor([0.31225, 0.59341, 0.74231], dtype=torch.float64)
        aux = aux / aux.norm()
    d_gimbal = torch.stack([aux[0], aux[1], torch.zeros((), dtype=torch.float64)])
    d_gimbal = d_gimbal / d_gimbal.norm()
    f64 = dict(dtype=torch.float64)
    R = torch.stack([
        torch.tensor([0.0, 0.0, 0.0], **f64),
        torch.tensor([0.0, 0.0, 1.3], **f64),   # +-z edges with atom 0
        torch.tensor([0.0, 0.0, -1.1], **f64),  # +-z edges with atoms 0 and 1
        1.4 * d_gimbal,                         # eSCN-gimbal edges with atom 0
        torch.tensor([0.9, -0.8, 0.5], **f64),
    ])
    Z = torch.tensor([6, 1, 8, 7, 1])
    return batch_from(Z, R, torch.zeros(5, dtype=torch.long))


def _gimbal_errors(model):
    """(energy, force) symmetry errors for maps onto and off the gimbal structure."""
    batch = _gimbal_molecule()
    E, F = predict(model, batch)
    assert torch.isfinite(E).all() and torch.isfinite(F).all()
    errs = [symmetry_errors(model, batch, g)
            for g in (random_rotation(11), random_rotation(12), improper_elements()["mirror_x"])]
    # ...and the reverse direction: a generic structure (R @ g) mapped by g back
    # ONTO the gimbal structure R.
    g = random_rotation(13)
    errs.append(symmetry_errors(model, with_positions(batch, batch["R"] @ g), g))
    return errs


def test_gimbal_edges_energy(small_model):
    """Energies (hence the Wigner-D VALUES) are exact on the gimbal."""
    for e_err, _ in _gimbal_errors(small_model):
        assert e_err < POLE_TOL, e_err


def test_gimbal_edges_forces(small_model):
    """Forces are exactly equivariant on the gimbal too (pole-free eSCN frame)."""
    for _, f_err in _gimbal_errors(small_model):
        assert f_err < POLE_TOL, f_err


@pytest.mark.xfail(strict=True, reason=(
    "escn_frame='euler' (the direct Euler-angle evaluation of the frame): "
    "matrix_to_euler_ZYZ snaps beta to exactly 0/pi when |R33| > 1 - 1e-6 and that "
    "branch carries no gradient through beta, so forces on edges within ~1.5e-3 rad "
    "of the eSCN gimbal directions are not exactly equivariant."))
def test_gimbal_edges_forces_euler_frame():
    model = build_model(dtype=torch.float64, seed=0, escn_frame="euler")
    for _, f_err in _gimbal_errors(model):
        assert f_err < POLE_TOL, f_err


def test_euler_and_polefree_frames_agree():
    """Same weights, both frame evaluations: the same function up to round-off."""
    batch = make_batch(seed=21)
    e = {}
    for frame in ("polefree", "euler"):
        model = build_model(dtype=torch.float64, seed=5, escn_frame=frame)
        e[frame] = predict(model, batch)
    assert max_rel_err(e["euler"][0], e["polefree"][0]) < 1e-10
    assert max_rel_err(e["euler"][1], e["polefree"][1]) < 1e-10


def test_build_model_frame_default_and_override():
    """A config without ``escn_frame`` builds the pole-free frame; ``euler`` is opt-in."""
    from AtomTrick.trainer import Trainer
    assert Trainer.build_model({}).escn_frame == "polefree"
    assert Trainer.build_model({"escn_frame": "euler"}).escn_frame == "euler"


# ---------------------------------------------------------------------------
# Ablation variants used by configs/rmd17/ablations/*.yaml
# ---------------------------------------------------------------------------

#: name -> model overrides on top of the (small) final configuration.  The
#: SO(2) K/V variant is only SO(3)-equivariant and has its own test module.
O3_VARIANTS = {
    "final": {},
    "o3_kv": {"kv": "o3"},
    "softmax": {"attn_nonlin": "softmax"},
    "perdegree_ln": {"layernorm_mode": "default"},
    "lmax1": {"irreps_hidden": "8x0e+8x1o"},
    "lmax2": {"irreps_hidden": "8x0e+8x1o+8x2e"},
}


@pytest.mark.parametrize("variant", sorted(O3_VARIANTS))
def test_variant_is_o3_equivariant(variant, small_batch):
    model = build_model(dtype=torch.float64, seed=3, **O3_VARIANTS[variant])
    for g in (random_rotation(5), improper_elements()["rotoinversion"]):
        e_err, f_err = symmetry_errors(model, small_batch, g)
        assert e_err < TOL and f_err < TOL, (variant, e_err, f_err)


@pytest.mark.parametrize("variant", sorted(O3_VARIANTS) + ["so2_kv"])
def test_variant_trains(variant, small_batch):
    """Every parameter receives a finite gradient from the force+energy loss."""
    overrides = {"kv": "so2"} if variant == "so2_kv" else O3_VARIANTS[variant]
    model = build_model(dtype=torch.float64, seed=4, train=True, **overrides)
    out = model(small_batch)
    loss = out["F"].pow(2).mean() + out["E"].pow(2).mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, f"{variant}: parameters without a finite gradient: {missing[:5]}"


def test_float32_matches_float64():
    """The production dtype (fp32) agrees with fp64 to single-precision accuracy."""
    b64 = make_batch((6, 5), dtype=torch.float64, seed=1)
    b32 = {k: (v.float() if v.is_floating_point() else v) for k, v in b64.items()}
    m64 = build_model(dtype=torch.float64, seed=2)
    m32 = build_model(dtype=torch.float32, seed=2)
    m32.load_state_dict({k: v.float() if v.is_floating_point() else v
                         for k, v in m64.state_dict().items()})
    E64, F64 = predict(m64, b64)
    E32, F32 = predict(m32, b32)
    assert max_rel_err(E32.double(), E64) < 1e-4
    assert max_rel_err(F32.double(), F64) < 1e-3
