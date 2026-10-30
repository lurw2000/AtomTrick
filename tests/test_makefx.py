"""The compiled training path (``make_fx`` + ``torch.compile``) against eager.

``make_fx`` traces the forward AND the force backward ``autograd.grad(E, R,
create_graph=True)`` into one aten graph, so a training step is a single
``loss.backward()`` that ``torch.compile`` can handle.  Tracing a backward
records the autograd engine's internal ``aten.detach`` nodes; replayed, they
cut the second-order graph: forces stay exact while ``d L_F / d theta`` is
silently wrong.  Every run trained before 2026-07-29 had this bug; these tests
pin the fix (``_strip_detach``) by requiring the traced step's parameter
gradients to equal eager autograd's.

CUDA-only (the production path; Inductor-on-CPU is not what was validated).
Folds in the former ``_forcefix_check/{ref_capture,compare}.py``.
"""
import copy

import pytest
import torch

from common import build_model, make_batch, requires_cuda, to_device

pytestmark = [requires_cuda, pytest.mark.slow]

RHO = 0.999


def _batch(dtype):
    b = to_device(make_batch((6, 6, 6, 6), dtype=dtype, seed=5), "cuda")
    g = torch.Generator(device="cpu").manual_seed(1)
    b["E"] = torch.randn(4, generator=g, dtype=dtype).cuda()
    b["F"] = torch.randn(b["R"].shape, generator=g, dtype=dtype).cuda()
    return b


def _loss(pred, batch):
    return (RHO * (pred["F"] - batch["F"]).pow(2).mean()
            + (1 - RHO) * (pred["E"] - batch["E"]).pow(2).mean())


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def _eager_grads(model, batch):
    model.zero_grad(set_to_none=True)
    pred = model(batch)
    _loss(pred, batch).backward()
    return {k: v.detach() for k, v in pred.items()}, _grads(model)


def _compare(g_ref, g_test, tol, per_tensor=True):
    """Relative L2 gradient error, per parameter tensor (fp64: catches a wrong
    gradient on any single tensor, e.g. the radial-basis parameters the detach
    bug hit) or over the concatenated gradient (fp32, where tiny-gradient
    tensors are dominated by atomic-scatter round-off)."""
    if not per_tensor:
        a = torch.cat([g.reshape(-1) for g in g_ref.values()])
        b = torch.cat([g_test[n].reshape(-1) for n in g_ref])
        rel = float((a - b).norm() / a.norm())
        assert rel < tol, f"make_fx gradient differs from eager: global rel {rel:.2e}"
        return
    worst, where = 0.0, ""
    for n, a in g_ref.items():
        b = g_test[n]
        rel = float((a - b).norm() / a.norm().clamp_min(1e-30))
        if rel > worst:
            worst, where = rel, n
    assert worst < tol, f"make_fx parameter gradient differs from eager: rel {worst:.2e} at {where}"


@pytest.fixture(autouse=True)
def _no_tf32():
    """Compare like with like: full-precision fp32 matmuls for the duration."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-8), (torch.float32, 1e-4)])
def test_traced_graph_gradients_equal_eager(dtype, tol):
    """The raw make_fx graph (no Inductor): isolates the tracing itself."""
    from AtomTrick.backward.makefx_compiled import MakeFXCompiledStep

    model = build_model(dtype=dtype, seed=7, train=True, device="cuda")
    batch = _batch(dtype)
    pred_ref, g_ref = _eager_grads(model, batch)

    step = MakeFXCompiledStep(model, batch, dynamic=True, train_mode=True)
    assert step._n_complex == 0, "Wigner cache was cold when tracing (complex ops in graph)"
    model.zero_grad(set_to_none=True)
    E, F = step.gm(step._pvals, *step._unpack(batch))
    assert torch.allclose(F.detach(), pred_ref["F"], rtol=tol, atol=tol * float(pred_ref["F"].abs().max()))
    _loss({"E": E, "F": F}, batch).backward()
    _compare(g_ref, _grads(model), tol, per_tensor=(dtype == torch.float64))


def test_compiled_step_gradients_equal_eager():
    """The torch.compile'd graph that training actually runs (fp64: Inductor's
    re-association stays at round-off level, so any real discrepancy shows)."""
    from AtomTrick.backward.makefx_compiled import MakeFXCompiledStep

    model = build_model(dtype=torch.float64, seed=7, train=True, device="cuda")
    batch = _batch(torch.float64)
    _, g_ref = _eager_grads(model, batch)
    step = MakeFXCompiledStep(model, batch, dynamic=True, train_mode=True)
    model.zero_grad(set_to_none=True)
    _loss(step.step(batch), batch).backward()
    _compare(g_ref, _grads(model), 1e-7)


def test_trainer_makefx_step_matches_eager(monkeypatch, tmp_path):
    """Same check through ``Trainer`` with ``ATOMTRICK_MAKEFX=1`` (what train.py
    sets for ``--makefx on``); lr=0 so the optimizer step leaves the weights alone."""
    monkeypatch.setenv("ATOMTRICK_MAKEFX", "1")
    from AtomTrick.trainer import Trainer

    dtype = torch.float64
    model = build_model(dtype=dtype, seed=8, train=True, device="cuda")
    ref = copy.deepcopy(model)
    batch = _batch(dtype)
    _, g_ref = _eager_grads(ref, batch)

    trainer = Trainer(model, device=torch.device("cuda"), lr=0.0, weight_decay=0.0,
                      warmup_steps=0, grad_clip=0.0, rho_force=RHO, out_dir=str(tmp_path))
    assert trainer.use_makefx
    trainer._makefx_backward_step(batch)
    _compare(g_ref, _grads(trainer.model), 1e-7)
