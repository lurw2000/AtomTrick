"""Loading checkpoints written by older versions of this code.

Two storage changes were made since; a checkpoint from before them must still load
with ``strict=True``:

1. equivariant linears stored ONE flat ``weight`` vector (now a
   ``weight_blocks`` ParameterList; a load pre-hook splits the old tensor);
2. every derived index buffer (``_row_idx``, ``_expand_idx``, ``_head_perm``,
   ...) was persistent, i.e. saved in the state_dict (13x the size of the
   weights).  The release makes them ``persistent=False``; on load it checks
   each legacy copy against the constructed value, then drops it (a mismatch
   is a load error).  ``atomic_ref_energy`` and ``energy_bias`` carry real
   data and stay persistent.

The legacy files are synthesised from a current model, so these tests need no
download.  To additionally check a real checkpoint::

    ATOMTRICK_TEST_CKPT=/path/to/run_best.pt python -m pytest tests/test_checkpoint_compat.py

(``args.json`` is read from the checkpoint's directory, or from
``ATOMTRICK_TEST_ARGS``).
"""
import json
import os

import pytest
import torch

from common import build_model, make_batch, predict, random_rotation, symmetry_errors

PERSISTENT_BUFFERS = {"atomic_ref_energy", "energy_bias"}


def _is_block_linear(mod) -> bool:
    return (getattr(mod, "weight_blocks", None) is not None
            and hasattr(mod, "_block_shapes") and len(mod.weight_blocks) > 0)


def to_legacy(sd, model, prefix=""):
    """Rewrite a current-format state_dict (of ``model``, keys optionally under
    ``prefix``) into the older format: flat weights + persistent buffers."""
    legacy = dict(sd)
    for name, buf in model.named_buffers():
        legacy.setdefault(prefix + name, buf.detach().clone())
    n_flat = 0
    for mod_name, mod in model.named_modules():
        if not _is_block_linear(mod):
            continue
        p = prefix + (mod_name + "." if mod_name else "")
        keys = [f"{p}weight_blocks.{i}" for i in range(len(mod.weight_blocks))]
        legacy[p + "weight"] = torch.cat([legacy.pop(k).reshape(-1) for k in keys])
        n_flat += 1
    assert n_flat > 0, "no block-parameter linears found -- storage layout changed?"
    return legacy


@pytest.fixture
def batch():
    return make_batch((6, 5), dtype=torch.float64, seed=0)


@pytest.mark.parametrize("variant", [{}, {"kv": "so2"}, {"kv": "o3"}])
def test_legacy_state_dict_loads_strictly(batch, variant):
    src = build_model(seed=0, **variant)
    legacy = to_legacy(src.state_dict(), src)
    assert any(k.endswith(".weight") and "o2_" in k for k in legacy) or variant.get("kv") == "o3"
    dst = build_model(seed=1, **variant)            # different init
    result = dst.load_state_dict(legacy, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for (n, p_src), (_, p_dst) in zip(src.named_parameters(), dst.named_parameters()):
        assert torch.equal(p_src, p_dst), n
    E0, F0 = predict(src, batch)
    E1, F1 = predict(dst, batch)
    assert torch.equal(E0, E1) and torch.equal(F0, F1)


def test_checkpoint_without_energy_bias_loads(batch):
    """Checkpoints older than the ``energy_bias`` buffer load with a zero bias."""
    src = build_model(seed=0)
    legacy = to_legacy(src.state_dict(), src)
    legacy.pop("energy_bias")
    dst = build_model(seed=1)
    with torch.no_grad():
        dst.energy_bias.fill_(1.0)
    dst.load_state_dict(legacy, strict=True)
    assert torch.count_nonzero(dst.energy_bias) == 0


def test_unknown_keys_still_fail_strict_loading():
    """The legacy-key filter must only drop known derived buffers."""
    src = build_model(seed=0)
    legacy = to_legacy(src.state_dict(), src)
    legacy["layers.0.attn.definitely_not_a_key"] = torch.zeros(3)
    with pytest.raises(RuntimeError):
        build_model(seed=1).load_state_dict(legacy, strict=True)


def test_changed_legacy_index_buffer_is_rejected():
    """A saved index buffer that differs from the constructed one means the
    checkpoint was written with another index layout: refuse, even non-strict."""
    src = build_model(seed=0)
    legacy = to_legacy(src.state_dict(), src)
    key = next(k for k in legacy if k.endswith("._head_perm"))
    legacy[key] = legacy[key].flip(0)
    for strict in (True, False):
        with pytest.raises(RuntimeError, match="differs from the one this code builds"):
            build_model(seed=1).load_state_dict(legacy, strict=strict)


def test_new_checkpoints_store_only_learned_state():
    model = build_model(seed=0)
    params = {n for n, _ in model.named_parameters()}
    extra = set(model.state_dict()) - params
    assert extra == PERSISTENT_BUFFERS, (
        "state_dict should hold parameters + atomic_ref_energy/energy_bias only; "
        f"unexpected persistent buffers: {sorted(extra - PERSISTENT_BUFFERS)[:8]}")


def test_state_dict_roundtrip_through_disk(tmp_path, batch):
    src = build_model(seed=0)
    path = tmp_path / "best.pt"
    torch.save(src.state_dict(), path)
    dst = build_model(seed=1)
    dst.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    assert torch.equal(predict(src, batch)[1], predict(dst, batch)[1])


def test_legacy_full_checkpoint_resumes_through_trainer(tmp_path, batch):
    """A full (``*_final.pt``-style) checkpoint with legacy model + EMA state."""
    from AtomTrick.trainer import Trainer

    tr_batch = dict(batch)
    tr_batch["E"] = torch.randn(2, dtype=torch.float64)
    tr_batch["F"] = torch.randn_like(batch["R"])
    tr_a = Trainer(build_model(seed=0, train=True), ema_decay=0.99, lr=1e-2,
                   warmup_steps=0, out_dir=str(tmp_path / "a"))
    for _ in range(3):
        loss, _, _ = Trainer.combined_loss(tr_a.model(tr_batch), tr_batch, 0.999)
        tr_a.optimizer.zero_grad()
        loss.backward()
        tr_a.optimizer.step()
        tr_a._ema_update()
    path = tmp_path / "final.pt"
    tr_a.save_checkpoint(str(path))

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["model"] = to_legacy(ckpt["model"], tr_a.model)
    ckpt["ema_model"] = to_legacy(ckpt["ema_model"], tr_a._ema_model.module, prefix="module.")
    torch.save(ckpt, path)

    tr_b = Trainer(build_model(seed=1, train=True), ema_decay=0.99, lr=1e-2,
                   warmup_steps=0, out_dir=str(tmp_path / "b"))
    tr_b.load_checkpoint(str(path))
    for (n, pa), (_, pb) in zip(tr_a.model.named_parameters(), tr_b.model.named_parameters()):
        assert torch.equal(pa, pb), n
    for (n, pa), (_, pb) in zip(tr_a._ema_model.module.named_parameters(),
                                tr_b._ema_model.module.named_parameters()):
        assert torch.equal(pa, pb), f"EMA {n}"


@pytest.mark.parametrize("saved_lr", [2.5e-4, None])
def test_unloadable_optimizer_state_keeps_the_learning_rate(tmp_path, saved_lr):
    """Fallback resume (optimizer state with another parameter count, as written
    before the block-parameter refactor): the moments restart, the learning rate
    must not stay at the warmup start value (lr * 1e-8) for the rest of the run."""
    from AtomTrick.trainer import Trainer

    lr, warmup = 1e-3, 10
    tr_a = Trainer(build_model(seed=0, train=True), lr=lr, warmup_steps=warmup,
                   out_dir=str(tmp_path / "a"))
    tr_a.step = 1000                                  # well past warmup
    path = tmp_path / "final.pt"
    tr_a.save_checkpoint(str(path))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    group = ckpt["optimizer"]["param_groups"][0]
    group["params"].append(max(group["params"]) + 1)  # parameter count no longer matches
    if saved_lr is None:
        del group["lr"]                               # -> configured lr (warmup is over)
    else:
        group["lr"] = saved_lr                        # e.g. lowered by the plateau scheduler
    torch.save(ckpt, path)

    tr_b = Trainer(build_model(seed=1, train=True), lr=lr, warmup_steps=warmup,
                   out_dir=str(tmp_path / "b"))
    assert tr_b.optimizer.param_groups[0]["lr"] < 1e-9     # warmup start value
    tr_b.load_checkpoint(str(path))
    assert not tr_b.optimizer.state, "the optimizer state must not have been loaded"
    assert tr_b.step == 1000
    assert tr_b.optimizer.param_groups[0]["lr"] == pytest.approx(saved_lr or lr, rel=1e-12)


def test_checkpoint_from_another_optimizer_is_refused(tmp_path):
    """Optimizer state with another group layout and non-AdamW settings must not
    fall back silently to fresh AdamW moments (that would be a different run)."""
    from AtomTrick.trainer import Trainer

    tr_a = Trainer(build_model(seed=0, train=True), lr=1e-3, warmup_steps=0,
                   out_dir=str(tmp_path / "a"))
    path = tmp_path / "final.pt"
    tr_a.save_checkpoint(str(path))
    params = list(tr_a.model.parameters())
    other = torch.optim.SGD([{"params": params[:len(params) // 2]},
                             {"params": params[len(params) // 2:], "lr": 1e-4}],
                            lr=1e-3, momentum=0.9)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["optimizer"] = other.state_dict()
    torch.save(ckpt, path)

    tr_b = Trainer(build_model(seed=1, train=True), lr=1e-3, warmup_steps=0,
                   out_dir=str(tmp_path / "b"))
    with pytest.raises(ValueError, match="different optimizer"):
        tr_b.load_checkpoint(str(path))


# ---------------------------------------------------------------------------
# Optional: a real checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.skipif("ATOMTRICK_TEST_CKPT" not in os.environ,
                    reason="set ATOMTRICK_TEST_CKPT=<checkpoint.pt> to test a real checkpoint")
def test_real_checkpoint_loads_and_is_equivariant(tmp_path):
    from AtomTrick.trainer import Trainer

    path = os.environ["ATOMTRICK_TEST_CKPT"]
    args_path = os.environ.get("ATOMTRICK_TEST_ARGS",
                               os.path.join(os.path.dirname(path), "args.json"))
    with open(args_path) as f:
        cfg = json.load(f)
    cfg.update(device="cpu", dtype="float32", out_dir=str(tmp_path), wandb_project=None,
               _skip_save_args=True)
    model = Trainer.from_config(cfg).model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:            # full checkpoint: use EMA weights
        ema = ckpt.get("ema_model")
        sd = ({k[len("module."):]: v for k, v in ema.items() if k.startswith("module.")}
              if ema else ckpt["model"])
    else:                                                     # *_best.pt: raw EMA state_dict
        sd = ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()

    known = [z for z in (1, 6, 7, 8) if float(model.atomic_ref_energy[z]) != 0.0] or [1, 6]
    b = make_batch((9,), dtype=torch.float32, seed=0, elements=known)
    E, F = predict(model, b)
    assert torch.isfinite(E).all() and torch.isfinite(F).all()
    e_err, f_err = symmetry_errors(model, b, random_rotation(0, torch.float32))
    assert e_err < 1e-4 and f_err < 1e-3, (e_err, f_err)
