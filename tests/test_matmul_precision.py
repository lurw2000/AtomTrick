"""float32 matmul precision: the default and how evaluate.py resolves it (CPU, seconds).

* ``train.py`` and the three base configs default to ``highest`` (full fp32
  matmuls); ``Trainer.from_config`` without the key does too.
* ``evaluate.py`` scores a checkpoint with the precision it was trained with: an
  explicit ``--matmul_precision`` wins, else the value recorded in the run's
  ``args.json``; an ``args.json`` without one is scored with ``high`` (TF32).  The result record
  stores the precision used and its source; a YAML ``--config`` (resolved through
  ``train.py``) is labelled as such.

``torch.set_float32_matmul_precision`` is a global switch that also exists on CPU
(it only changes kernels on CUDA), so all of this is checked without a GPU.
"""
import json
import os

import pytest
import torch

from common import REPO_ROOT, write_npz

import evaluate  # noqa: E402  (repo root is on sys.path via conftest)

SMALL_CFG = {
    "irreps": "8x0e+8x1o+8x2e", "n_heads": 2, "n_layers": 1, "num_radial_basis": 8,
    "radial_hidden": 16, "energy_hidden": 16, "max_z": 10, "cutoff": 5.0,
    "edge_cutoff": 5.0,
}


@pytest.fixture(autouse=True)
def _restore_matmul_precision():
    prev = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(prev)


@pytest.mark.parametrize("configs", [
    [],
    ["configs/rmd17/atomtrick.yaml"],
    ["configs/md22/atomtrick.yaml"],
    ["configs/spice/stage1_force.yaml"],
])
def test_train_default_is_full_fp32(configs):
    import train
    argv = ["--dry_run", "--molecule", "aspirin"]
    for c in configs:
        argv += ["--config", os.path.join(REPO_ROOT, c)]
    _parser, args = train.parse_args(argv)
    assert args.matmul_precision == "highest"


def test_trainer_from_config_default_is_full_fp32(tmp_path):
    from AtomTrick.trainer import Trainer
    torch.set_float32_matmul_precision("high")
    Trainer.from_config(dict(SMALL_CFG, device="cpu", dtype="float32",
                             out_dir=str(tmp_path), _skip_save_args=True))
    assert torch.get_float32_matmul_precision() == "highest"


@pytest.mark.parametrize("cfg, requested, expected", [
    ({}, None, ("high", "not recorded")),                                   # no key recorded
    ({"matmul_precision": "highest"}, None, ("highest", "training config")),
    ({"matmul_precision": "high"}, None, ("high", "training config")),
    ({"_provenance": {"matmul_precision": "highest"}}, None,
     ("highest", "training config (_provenance)")),
    ({"matmul_precision": "highest"}, "high", ("high", "explicit")),       # flag wins
    ({}, "highest", ("highest", "explicit")),
])
def test_resolve_matmul_precision(cfg, requested, expected):
    prec, source = evaluate.resolve_matmul_precision(cfg, requested)
    assert prec == expected[0]
    assert source.startswith(expected[1])


def _train_tiny(tmp_path, name, **extra):
    from AtomTrick.trainer import Trainer
    npz = write_npz(tmp_path / "tiny.npz", n_frames=6)
    out = tmp_path / name
    cfg = dict(SMALL_CFG, data=npz, val_data=npz, warmup_steps=0, eval_every=2,
               save_every=2, batch_size=2, eval_batch_size=3, num_workers=0, seed=0,
               device="cpu", dtype="float32", out_dir=str(out), wandb_name=name,
               wandb_project=None, **extra)
    trainer = Trainer.from_config(cfg)
    train_loader, val_loader, _ = Trainer.build_loaders(cfg)
    trainer.fit(2, train_loader, val_loader)
    return npz, out / f"{name}_best.pt", out / "args.json"


def test_evaluate_uses_the_recorded_precision(tmp_path):
    # A run of this code records its precision; a config without the key gets TF32.
    npz, ckpt_new, _ = _train_tiny(tmp_path, "new", matmul_precision="highest",
                                   _provenance={"matmul_precision": "highest"})
    npz, ckpt_old, args_old = _train_tiny(tmp_path, "old")
    saved = json.loads(args_old.read_text())
    assert "matmul_precision" not in saved
    assert "matmul_precision" not in saved["_provenance"]

    def run(ckpt, **kw):
        rec = evaluate.evaluate_checkpoint(str(ckpt), npz, device="cpu", eval_bs=3, **kw)
        assert torch.get_float32_matmul_precision() == rec["eval"]["matmul_precision"]
        return rec["eval"]["matmul_precision"], rec["eval"]["matmul_precision_source"]

    prec, src = run(ckpt_new)
    assert prec == "highest" and src.startswith("training config")
    prec, src = run(ckpt_old)
    assert prec == "high" and src.startswith("not recorded")
    prec, src = run(ckpt_new, matmul_precision="high")
    assert (prec, src) == ("high", "explicit")
    prec, src = run(ckpt_old, matmul_precision="highest")
    assert (prec, src) == ("highest", "explicit")


def test_evaluate_yaml_config_route_is_labelled(tmp_path, monkeypatch):
    # A YAML --config is resolved through train.py, which always fills
    # matmul_precision (YAML value, else its 'highest' default): the record must say
    # where the value came from instead of claiming the run recorded it.
    npz, ckpt, args_json = _train_tiny(tmp_path, "yaml", matmul_precision="highest")
    resolved = evaluate.load_config_files([str(args_json)])
    calls = []

    def fake_resolve_train_yaml(paths, molecule=None):
        calls.append(list(paths))
        return dict(resolved, matmul_precision="highest")

    monkeypatch.setattr(evaluate, "resolve_train_yaml", fake_resolve_train_yaml)
    rec = evaluate.evaluate_checkpoint(str(ckpt), npz, config_paths=["model.yaml"],
                                       device="cpu", eval_bs=3)
    assert calls == [["model.yaml"]]
    assert rec["eval"]["matmul_precision"] == "highest"
    assert rec["eval"]["matmul_precision_source"].startswith("YAML --config")
    rec = evaluate.evaluate_checkpoint(str(ckpt), npz, config_paths=["model.yaml"],
                                       device="cpu", eval_bs=3, matmul_precision="high")
    assert (rec["eval"]["matmul_precision"], rec["eval"]["matmul_precision_source"]) \
        == ("high", "explicit")


def test_evaluate_cli_default_is_unset():
    args = evaluate.build_parser().parse_args(["--checkpoint", "c.pt", "--data", "d.npz"])
    assert args.matmul_precision is None
