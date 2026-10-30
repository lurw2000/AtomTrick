"""Trainer / train.py guards that only matter on the compiled path or on resume.

* ``eval_batch_size``: the compiled graph has the training molecule count baked
  in, so ``train.py`` clamps a larger eval batch up front instead of failing at
  the first validation.
* SWA resume: a checkpoint written inside the SWA stage holds a UNIFORM weight
  average; resuming it must keep averaging uniformly, not switch to the EMA the
  trainer builds by default.  A checkpoint at exactly ``step == swa_start``
  predates the switch.

All CPU; the make_fx graph itself is never built here.
"""
import argparse
import os

import pytest
import torch

from AtomTrick.trainer import Trainer

from common import build_model, make_batch


def _batch():
    b = make_batch((6,), dtype=torch.float32, seed=0)
    g = torch.Generator().manual_seed(1)
    b["E"] = torch.randn(1, generator=g)
    b["F"] = torch.randn(b["R"].shape, generator=g)
    return b


def _model(seed=0, **kw):
    return build_model(dtype=torch.float32, seed=seed, n_layers=1, train=True, **kw)


def _train(trainer, batch, steps):
    for _ in range(steps):
        loss, _, _ = Trainer.combined_loss(trainer.model(batch), batch, trainer.rho_force)
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        trainer._ema_update()
        trainer.step += 1


def _is_uniform(avg_model):
    return avg_model.avg_fn is None and avg_model.multi_avg_fn is None


# ---------------------------------------------------------------------------
# eval_batch_size clamp (train.py)
# ---------------------------------------------------------------------------

def _ns(**kw):
    base = dict(batch_size=2, eval_batch_size=4)
    base.update(kw)
    return argparse.Namespace(**base)


def test_eval_batch_size_is_clamped_on_the_makefx_path():
    import train
    a = _ns()
    train.clamp_eval_batch_size(a, makefx=True)
    assert a.eval_batch_size == 2
    a = _ns(eval_batch_size=None, batch_size=4)    # None = batch_size: fine
    train.clamp_eval_batch_size(a, makefx=True)
    assert a.eval_batch_size is None


@pytest.mark.parametrize("kw, makefx", [({}, False), ({"eval_batch_size": 1}, True)])
def test_eval_batch_size_left_alone(kw, makefx):
    import train
    a = _ns(**kw)
    before = a.eval_batch_size
    train.clamp_eval_batch_size(a, makefx=makefx)
    assert a.eval_batch_size == before


# ---------------------------------------------------------------------------
# SWA resume
# ---------------------------------------------------------------------------

def _swa_trainer(tmp_path, seed=0):
    return Trainer(_model(seed=seed), ema_decay=0.99, lr=1e-2, warmup_steps=0,
                   out_dir=str(tmp_path), swa_start=5, swa_rho_force=0.1, rho_force=0.9)


def test_resume_inside_swa_stage_keeps_uniform_average(tmp_path):
    batch = _batch()
    t = _swa_trainer(tmp_path)
    _train(t, batch, 5)
    t._enter_swa_stage()                    # what fit() does at step == swa_start
    assert _is_uniform(t._ema_model)
    _train(t, batch, 3)                     # step 8, inside the SWA stage
    path = os.path.join(str(tmp_path), "rolling.pt")
    t.save_checkpoint(path)

    t2 = _swa_trainer(tmp_path, seed=1)
    assert not _is_uniform(t2._ema_model)   # EMA by default
    t2.load_checkpoint(path)
    assert _is_uniform(t2._ema_model)
    assert t2._in_swa_stage(t2.step)
    for (k, v), (_, w) in zip(t._ema_model.state_dict().items(),
                              t2._ema_model.state_dict().items()):
        assert torch.equal(v, w), k
    # both keep averaging the same way
    t2.rho_force = t.rho_force
    _train(t, batch, 2)
    _train(t2, batch, 2)
    for (k, v), (_, w) in zip(t._ema_model.state_dict().items(),
                              t2._ema_model.state_dict().items()):
        assert torch.allclose(v.double(), w.double(), rtol=1e-5, atol=1e-7), k


def test_checkpoint_at_swa_start_predates_the_switch(tmp_path):
    batch = _batch()
    t = _swa_trainer(tmp_path)
    _train(t, batch, 5)                     # saved before fit() would switch
    path = os.path.join(str(tmp_path), "at_start.pt")
    t.save_checkpoint(path)
    t2 = _swa_trainer(tmp_path, seed=1)
    t2.load_checkpoint(path)
    assert t2.step == 5 and not t2._in_swa_stage(t2.step)
    assert not _is_uniform(t2._ema_model)   # still the stage-1 EMA
