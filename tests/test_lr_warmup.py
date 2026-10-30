"""Linear learning-rate warmup (10,000 steps in the paper runs; 20 here)."""
import pytest
import torch

from AtomTrick.trainer import Trainer

from common import build_model, make_batch


def test_linear_warmup_reaches_base_lr(tmp_path):
    base_lr, warmup = 1e-3, 20
    model = build_model(dtype=torch.float32, seed=0, n_layers=1, train=True)
    trainer = Trainer(model, lr=base_lr, warmup_steps=warmup, eval_every=10 ** 6,
                      out_dir=str(tmp_path))
    batch = make_batch((5,), dtype=torch.float32, seed=0)
    batch["E"] = torch.tensor([-5.0])
    batch["F"] = torch.randn(batch["R"].shape)

    lrs = []
    for _ in range(warmup + 5):
        loss, _, _ = Trainer.combined_loss(model(batch), batch, 0.5)
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainer.optimizer.step()
        # the same stepping rule as Trainer.fit
        if trainer._warmup_scheduler is not None and trainer.step < trainer.warmup_steps:
            trainer._warmup_scheduler.step()
        trainer.step += 1
        lrs.append(trainer.optimizer.param_groups[0]["lr"])

    assert lrs[0] < 0.1 * base_lr, "warmup must start near zero"
    assert all(b >= a for a, b in zip(lrs[:warmup], lrs[1:warmup + 1])), "LR must rise monotonically"
    assert lrs[warmup // 2 - 1] == pytest.approx(0.5 * base_lr, rel=0.1)
    assert lrs[warmup - 1] == pytest.approx(base_lr, rel=1e-6)
    assert lrs[-1] == pytest.approx(base_lr, rel=1e-6), "LR must stay at base after warmup"
