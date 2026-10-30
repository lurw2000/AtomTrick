"""EMA of the weights in ``Trainer`` (decay 0.99 in every paper run).

Validation, ``*_best.pt`` and all evaluation use the EMA weights, so the swap
in/out logic and its checkpointing must be exact.
"""
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


def _model(seed=0):
    return build_model(dtype=torch.float32, seed=seed, n_layers=1, train=True)


def _train(trainer, batch, steps=10):
    for _ in range(steps):
        loss, _, _ = Trainer.combined_loss(trainer.model(batch), batch, 0.5)
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        trainer._ema_update()
        trainer.step += 1


@pytest.fixture
def trainer(tmp_path):
    return Trainer(_model(), ema_decay=0.99, lr=1e-2, warmup_steps=0, out_dir=str(tmp_path))


def test_ema_created(trainer):
    assert trainer._ema_model is not None


def test_ema_not_created_when_zero(tmp_path):
    t = Trainer(_model(), ema_decay=0.0, warmup_steps=0, out_dir=str(tmp_path))
    assert t._ema_model is None


def test_ema_weights_lag_the_training_weights(trainer):
    _train(trainer, _batch())
    ema = trainer._ema_model.module.state_dict()
    live = trainer.model.state_dict()
    assert any(not torch.equal(live[k], ema[k]) for k in live)


def test_ema_apply_and_restore_swap_exactly(trainer):
    _train(trainer, _batch())
    live = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    ema = {k: v.clone() for k, v in trainer._ema_model.module.state_dict().items()}
    trainer._ema_apply()
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v, ema[k]), k
    trainer._ema_restore()
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v, live[k]), k


def test_ema_changes_predictions(trainer):
    batch = _batch()
    _train(trainer, batch)
    trainer._ema_apply()
    trainer.model.eval()
    e_ema = trainer.model(batch)["E"].clone()
    trainer.model.train()
    trainer._ema_restore()
    trainer.model.eval()
    e_live = trainer.model(batch)["E"].clone()
    assert not torch.equal(e_ema, e_live)


def test_ema_checkpoint_roundtrip(trainer, tmp_path):
    _train(trainer, _batch())
    ema_before = {k: v.clone() for k, v in trainer._ema_model.module.state_dict().items()}
    path = os.path.join(str(tmp_path), "ckpt.pt")
    trainer.save_checkpoint(path)
    t2 = Trainer(_model(seed=1), ema_decay=0.99, lr=1e-2, warmup_steps=0,
                 out_dir=str(tmp_path))
    t2.load_checkpoint(path)
    for k, v in t2._ema_model.module.state_dict().items():
        assert torch.equal(v, ema_before[k]), k
    assert t2.step == 10


def test_ema_helpers_are_noops_when_disabled(tmp_path):
    t = Trainer(_model(), ema_decay=0.0, warmup_steps=0, out_dir=str(tmp_path))
    before = {k: v.clone() for k, v in t.model.state_dict().items()}
    t._ema_update()
    t._ema_apply()
    t._ema_restore()
    for k, v in t.model.state_dict().items():
        assert torch.equal(v, before[k])
