"""End-to-end smoke tests on a tiny synthetic dataset (CPU, seconds).

* ``Trainer.from_config`` -> ``build_loaders`` -> ``fit`` writes the documented
  checkpoint files, and ``*_best.pt`` reloads strictly into a fresh model;
* ``train.py`` run as a user would, then ``evaluate.py`` on its checkpoint;
  the metrics ``evaluate.py`` reports are recomputed here from their definitions
  (pooled over the whole set, float64 accumulation).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from common import REPO_ROOT, run_train, write_npz

SMALL_CFG = {
    "irreps": "8x0e+8x1o+8x2e+8x3o", "n_heads": 2, "n_layers": 1, "ff_mult": 4,
    "num_radial_basis": 8, "radial_hidden": 16, "energy_hidden": 16, "max_z": 10,
    "cutoff": 5.0, "edge_cutoff": 5.0, "radial_mode": "gate", "rbf_type": "expnorm",
    "k_mode": "o2", "v_mode": "o2", "ffn_act_mode": "gate",
    "attn_nonlin": "identity", "layernorm_mode": "shared",
}
SMALL_CLI = ["--irreps", "8x0e+8x1o+8x2e+8x3o", "--n_heads", "2", "--n_layers", "1",
             "--num_radial_basis", "8", "--radial_hidden", "16", "--energy_hidden", "16",
             "--max_z", "10"]


def test_trainer_fit_writes_checkpoints(tmp_path):
    from AtomTrick.trainer import Trainer

    npz = write_npz(tmp_path / "tiny.npz", n_frames=12)
    out = tmp_path / "run"
    cfg = dict(SMALL_CFG, data=npz, val_data=npz, lr=1e-3, weight_decay=0.0, rho_force=0.9,
               warmup_steps=2, ema_decay=0.99, eval_every=5, save_every=5,
               batch_size=2, eval_batch_size=5, num_workers=0,
               seed=0, device="cpu", dtype="float32", out_dir=str(out),
               wandb_name="smoke", wandb_project=None)
    trainer = Trainer.from_config(cfg)
    train_loader, val_loader, _ = Trainer.build_loaders(cfg)
    trainer.fit(10, train_loader, val_loader)
    assert trainer.step == 10
    for name in ("smoke_best.pt", "smoke_final.pt", "smoke.pt", "args.json"):
        assert (out / name).exists(), f"missing {name}"

    fresh = Trainer.from_config(dict(cfg, out_dir=str(tmp_path / "reload"), seed=123)).model
    fresh.load_state_dict(torch.load(out / "smoke_best.pt", map_location="cpu",
                                     weights_only=True), strict=True)
    full = torch.load(out / "smoke_final.pt", map_location="cpu", weights_only=False)
    assert full["step"] == 10 and {"model", "optimizer", "ema_model"} <= set(full)


def _find_metric_dicts(obj):
    """All dicts in a JSON tree that carry evaluate.py's force metric."""
    if isinstance(obj, dict):
        if "f_mae_meVA" in obj:
            yield obj
        for v in obj.values():
            yield from _find_metric_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find_metric_dicts(v)


@pytest.mark.slow
def test_train_py_then_evaluate_py(tmp_path):
    npz = write_npz(tmp_path / "tiny.npz", n_frames=12)
    res = run_train(["--data", npz, "--val_data", npz, "--out_dir", "runs/smoke",
                     "--wandb_name", "smoke", *SMALL_CLI, "--max_iterations", "10",
                     "--eval_every", "5", "--save_every", "5", "--batch_size", "2",
                     "--eval_batch_size", "2", "--warmup_steps", "2", "--num_workers", "0",
                     "--makefx", "off"], cwd=str(tmp_path))
    assert res.returncode == 0, res.stdout[-4000:]
    run = tmp_path / "runs" / "smoke"
    for name in ("smoke_best.pt", "smoke_final.pt", "args.json"):
        assert (run / name).exists(), f"missing {name}"
    args = json.loads((run / "args.json").read_text())
    assert "_provenance" in args, "args.json should record code/torch provenance"
    assert args.get("n_layers") == 1 and args.get("attn_nonlin") == "identity"

    evaluate = os.path.join(REPO_ROOT, "evaluate.py")
    if not os.path.exists(evaluate):
        pytest.skip("evaluate.py not present")
    out_json = tmp_path / "metrics.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ATOMTRICK_", "ATOMTRICKS_"))}
    env["CUDA_VISIBLE_DEVICES"] = ""
    ev = subprocess.run([sys.executable, evaluate, "--checkpoint", str(run / "smoke_best.pt"),
                         "--data", npz, "--device", "cpu", "--makefx", "off",
                         "--eval_bs", "5", "--out", str(out_json)],
                        cwd=str(tmp_path), env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, timeout=600)
    assert ev.returncode == 0, ev.stdout[-4000:]
    rows = list(_find_metric_dicts(json.loads(out_json.read_text())))
    assert rows, f"no metrics in {out_json}"
    got = rows[0]

    # Recompute the documented metrics from scratch.
    from AtomTrick.data import MLFFDataset, collate_fn
    from AtomTrick.trainer import Trainer
    cfg = dict(args, device="cpu", out_dir=str(tmp_path / "ref"), wandb_project=None,
               _skip_save_args=True)
    model = Trainer.from_config(cfg).model
    model.load_state_dict(torch.load(run / "smoke_best.pt", map_location="cpu",
                                     weights_only=True), strict=True)
    model.eval()
    ds = MLFFDataset(npz, edge_cutoff=5.0)
    batch = collate_fn([ds[i] for i in range(len(ds))])
    pred = model(batch)
    dE = (pred["E"].double() - batch["E"].double()).numpy()
    dF = (pred["F"].double() - batch["F"].double()).numpy()
    n_at = batch["N"].double().numpy()
    ref = {
        "e_mae_meV": 1e3 * np.abs(dE).mean(),
        "e_rmse_meV": 1e3 * np.sqrt((dE ** 2).mean()),
        "e_mae_meV_atom": 1e3 * np.abs(dE / n_at).mean(),
        "f_mae_meVA": 1e3 * np.abs(dF).mean(),
        "f_rmse_meVA": 1e3 * np.sqrt((dF ** 2).mean()),
        "f_l2_meVA": 1e3 * np.linalg.norm(dF, axis=1).mean(),
    }
    assert int(got["n_structures"]) == len(ds)
    assert int(got["n_atoms"]) == int(n_at.sum())
    # --eval_bs 5 over 12 structures leaves a runt batch of 2: a batch-averaged
    # (instead of pooled) metric would miss these tolerances by far.  Energies
    # get a looser tolerance in case the tool compares total (reference-added)
    # energies, whose float32 representation is only good to ~1 meV.
    for key, value in ref.items():
        assert key in got, f"evaluate.py output lacks {key}"
        rel = 2e-3 if key.startswith("e_") else 1e-4
        assert got[key] == pytest.approx(value, rel=rel, abs=1e-6), key
