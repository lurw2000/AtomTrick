"""Forces and training gradients against central finite differences (fp64).

Two independent referees:

1. ``F = -dE/dR`` (the autograd force) against central differences of the
   energy, over every Cartesian coordinate.
2. The *training* gradient ``d L_F / d theta`` of the force loss -- the second-
   order quantity a force-matching run actually follows -- against central
   differences of the loss value.  This is the check that exposed the make_fx
   ``aten.detach`` bug (fixed 2026-07-29): forces were exact, while
   ``d L_F / d theta`` was ~4% wrong, worst on the radial-basis parameters
   ``rbf.means`` / ``rbf.betas``.  Here it guards the eager path; the make_fx
   path is compared against eager in ``test_makefx.py``.

Folds in the former ``_forcefix_check/diag_fd_referee.py`` and the
finite-difference half of ``_forcefix_check/ref_capture.py``.
"""
import pytest
import torch

from common import build_model, make_batch, max_rel_err, predict, with_positions

H_POS = 1e-5     # displacement for dE/dR  (truncation ~H^2, round-off ~1e-16/H)
H_PAR = 1e-6     # displacement for dL/dtheta


@pytest.fixture(scope="module")
def model():
    return build_model(dtype=torch.float64, seed=10)


@pytest.fixture(scope="module")
def batch():
    return make_batch((5, 4), dtype=torch.float64, seed=3)


def test_forces_match_central_differences(model, batch):
    _, F = predict(model, batch)
    R0 = batch["R"]
    F_fd = torch.zeros_like(R0)
    for i in range(R0.shape[0]):
        for k in range(3):
            E = []
            for sgn in (+1.0, -1.0):
                R = R0.clone()
                R[i, k] += sgn * H_POS
                E.append(predict(model, with_positions(batch, R))[0].sum())
            F_fd[i, k] = -(E[0] - E[1]) / (2 * H_POS)
    err = max_rel_err(F, F_fd)
    assert err < 1e-6, f"autograd forces disagree with finite differences: rel err {err:.2e}"


def test_train_mode_gives_the_same_forces(batch):
    """create_graph=True (training) must not change the force values."""
    m = build_model(dtype=torch.float64, seed=10)
    _, F_eval = predict(m, batch)
    m.train()
    F_train = m(batch)["F"].detach()
    assert max_rel_err(F_train, F_eval) < 1e-12


def _force_loss(model, batch, F_target):
    return (model(batch)["F"] - F_target).pow(2).mean()


def test_force_loss_parameter_gradient_matches_finite_differences(batch):
    model = build_model(dtype=torch.float64, seed=11, train=True)
    g = torch.Generator().manual_seed(0)
    F_target = torch.randn(batch["R"].shape, generator=g, dtype=torch.float64)

    model.zero_grad(set_to_none=True)
    _force_loss(model, batch, F_target).backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    params = dict(model.named_parameters())

    # Probe the parameters the make_fx bug hit hardest (radial basis) plus one
    # tensor of every other kind on the force path.
    wanted = ["rbf.means", "rbf.betas", "mlp_k.", "mlp_v.", "o2_k.", "o2_v.", "W_q.",
              "W_o.", "ffn_up.", "ffn_down.", "norm1.", "energy_proj.", "atom_embed."]
    probe = []
    for key in wanted:
        names = [n for n in params if key in n and "layers.1." in n] or \
                [n for n in params if key in n]
        assert names, f"no parameter matching {key!r}"
        probe.append(names[0])

    worst = (0.0, "")
    for name in probe:
        p = params[name]
        flat = p.data.view(-1)
        gflat = grads[name].view(-1)
        # the three largest-gradient entries of this tensor (well above FD noise)
        idxs = torch.topk(gflat.abs(), k=min(3, gflat.numel())).indices.tolist()
        for i in idxs:
            orig = flat[i].item()
            with torch.no_grad():
                flat[i] = orig + H_PAR
                lp = _force_loss(model, batch, F_target).item()
                flat[i] = orig - H_PAR
                lm = _force_loss(model, batch, F_target).item()
                flat[i] = orig
            fd = (lp - lm) / (2 * H_PAR)
            an = gflat[i].item()
            rel = abs(an - fd) / max(abs(fd), 1e-8)
            if rel > worst[0]:
                worst = (rel, f"{name}[{i}] analytic={an:.6e} fd={fd:.6e}")
    assert worst[0] < 1e-4, f"force-loss gradient disagrees with finite differences: {worst[1]}"
