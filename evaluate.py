"""Evaluate trained AtomTrick checkpoints: energy and force errors.

Replaces the former ``test.py``, ``eval_mace_metrics.py`` and
``eval_spice_subsets.py`` with one script and one set of metric definitions.

Metrics (float64 accumulation, pooled over the whole evaluation set; never a
mean of per-batch means):

    e_mae_meV / e_rmse_meV            total-energy error per structure (meV)
    e_mae_meV_atom / e_rmse_meV_atom  per-structure energy error / n_atoms (meV/atom)
    f_mae_meVA / f_rmse_meVA          per Cartesian force component (meV/Å)
    f_l2_meVA                         mean over atoms of ||F_pred - F_true||_2 (meV/Å)
    n_structures, n_atoms

``f_mae_batchavg_meVA`` is also reported.  It is the metric the trainer logs
during training (mean over batches of the per-batch component MAE); it depends
on ``--eval_bs`` and exists only so logged validation numbers can be reproduced.
Do not report it in papers; use ``f_mae_meVA``.

With ``--per_subset`` and an NPZ that carries ``subset`` (and ``subset_names``),
the same metrics are also computed per subset (SPICE: PubChem, DES370K Monomers,
DES370K Dimers, Dipeptides, Solvated Amino Acids, Water, QMugs).

Checkpoints
-----------
* ``*_best.pt``: a raw ``state_dict`` saved while the EMA (or SWA-average)
  weights were swapped in, so it IS the EMA snapshot.  Loaded as is.
* ``*_final.pt`` / rolling ``<name>.pt``: full training dict.  The EMA weights
  (``ema_model``) are evaluated, as the trainer's validation loop does.  If the
  dict has no EMA state, the raw training weights are used (with a warning).

The model configuration is read from ``args.json`` next to the checkpoint,
unless one or more ``--config`` files (JSON or YAML, merged in order) are given.

Labels
------
Energies/forces are converted and referenced in float64 and only then cast to
the model dtype (``label_mode: float64`` in every result record).

Matmul precision
----------------
float32 models are evaluated with the matmul precision they were trained with:
``--matmul_precision`` if given, else the value train.py recorded in the run's
``args.json`` (``_provenance`` / ``matmul_precision``; release default
``highest`` = full fp32).  A config that records none (written by an older
version of this code, which trained with TF32 matmuls) is scored with ``high``
(TF32).  The precision used and where it came from
are stored in every result record (``eval.matmul_precision``,
``eval.matmul_precision_source``).

Examples
--------
    # One checkpoint, one test set
    python evaluate.py --checkpoint runs/rmd17_atomtrick/aspirin/s42/rmd17_atomtrick_aspirin_s42_best.pt \\
        --data data/rmd17/aspirin_test_default.npz

    # All rMD17 molecules and three seeds, JSON + CSV output
    python evaluate.py \\
        --checkpoint "runs/rmd17_atomtrick/{molecule}/s{seed}/rmd17_atomtrick_{molecule}_s{seed}_best.pt" \\
        --data "data/rmd17/{molecule}_test_default.npz" \\
        --molecules aspirin azobenzene benzene ethanol malonaldehyde naphthalene \\
                    paracetamol salicylic toluene uracil \\
        --seeds 42 1 2 --tag atomtrick --out results/rmd17_test.json --csv results/rmd17_test.csv

    # SPICE with the per-subset breakdown
    python evaluate.py --checkpoint runs/spice_stage2_swa/s42/spice_stage2_swa_s42_best.pt \\
        --data data/spice_maceoff/spice_maceoff_test_f15.npz --per_subset --eval_bs 16

    # Quick smoke test on the first 16 structures
    python evaluate.py --checkpoint CKPT --data NPZ --limit 16
"""

import argparse
import csv
import datetime
import functools
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import traceback

import numpy as np

logger = logging.getLogger("evaluate")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

METRIC_KEYS = [
    "e_mae_meV", "e_rmse_meV", "e_mae_meV_atom", "e_rmse_meV_atom",
    "f_mae_meVA", "f_rmse_meVA", "f_l2_meVA",
]

# Model-config keys copied into each result record, so that aggregation and
# ablation analysis never have to go back to args.json.
MODEL_KEYS = [
    "irreps", "n_layers", "n_heads", "k_mode", "v_mode", "attn_nonlin",
    "layernorm_mode", "ffn_act_mode",
    "radial_mode", "rbf_type", "num_radial_basis", "cutoff", "dtype",
    "optimizer", "seed",
]

# Canonical SPICE subset names (MACE-OFF23 Table S1 spelling).  The NPZ stores
# the MACE-OFF23 xyz ``config_type`` strings (e.g. "water", "DES370K Dimers").
SPICE_SUBSET_ORDER = [
    "PubChem", "DES370K Monomers", "DES370K Dimers", "Dipeptides",
    "Solvated Amino Acids", "Water", "QMugs",
]


def canonical_subset_name(name):
    """Map a raw ``subset_names`` entry to its canonical display name."""
    n = str(name).lower()
    if "pubchem" in n:
        return "PubChem"
    if "dipeptide" in n:          # check before "des": "... Dipeptides" contains "des"
        return "Dipeptides"
    if "monomer" in n:
        return "DES370K Monomers"
    if "dimer" in n:
        return "DES370K Dimers"
    if "solvated" in n or "amino" in n:
        return "Solvated Amino Acids"
    if "water" in n:
        return "Water"
    if "qmug" in n:
        return "QMugs"
    return str(name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate AtomTrick checkpoints (energy/force MAE, RMSE, L2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n--------\n", 1)[-1])
    p.add_argument("--checkpoint", nargs="+", action="extend", required=True,
                   help="Checkpoint path(s). '{molecule}' / '{seed}' are expanded "
                        "over --molecules / --seeds.")
    p.add_argument("--data", nargs="+", action="extend", required=True,
                   help="NPZ path(s): one (used for every checkpoint) or one per "
                        "--checkpoint. '{molecule}' / '{seed}' are expanded.")
    p.add_argument("--config", action="append", default=None,
                   help="Model config (args.json or YAML) used INSTEAD of the "
                        "args.json next to the checkpoint. May be repeated; later "
                        "files override earlier ones. Placeholders are expanded. "
                        "A YAML config is resolved through train.py, so its "
                        "matmul_precision (default 'highest') applies instead of "
                        "the checkpoint's recorded precision; pass --matmul_precision "
                        "to override.")
    p.add_argument("--molecules", nargs="+", default=None,
                   help="Values substituted for '{molecule}'.")
    p.add_argument("--seeds", nargs="+", default=None,
                   help="Values substituted for '{seed}'.")
    p.add_argument("--tag", default=None,
                   help="Label stored in every result record (used by "
                        "scripts/aggregate/*). Default: the config's 'experiment' "
                        "value, else 'default'.")
    p.add_argument("--per_subset", action="store_true",
                   help="Also report metrics per subset (NPZ must have 'subset').")
    p.add_argument("--eval_bs", type=int, default=8,
                   help="Evaluation batch size (structures). Eager mode halves it "
                        "automatically on CUDA out-of-memory. Default 8.")
    p.add_argument("--device", default=None,
                   help="cuda | cuda:N | cpu (default: cuda if available).")
    p.add_argument("--makefx", choices=["off", "on"], default="off",
                   help="off (default): eager autograd forces. on: the compiled "
                        "make_fx + torch.compile inference graph (faster on large "
                        "sets, minutes of compile per checkpoint).")
    p.add_argument("--matmul_precision", choices=["highest", "high"], default=None,
                   help="torch float32 matmul precision ('highest' = full fp32, "
                        "'high' = TF32). Default: the precision the checkpoint was "
                        "trained with (its args.json); 'high' if it records none.")
    p.add_argument("--escn_frame", choices=["polefree", "euler"], default=None,
                   help="Override the eSCN-frame evaluation. Default: the checkpoint's "
                        "own (its args.json; missing key = 'polefree').")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N structures (smoke tests).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (default 0).")
    p.add_argument("--out", default=None,
                   help="Write the list of result records to this JSON file "
                        "(rewritten after every checkpoint).")
    p.add_argument("--csv", default=None,
                   help="Write a flat CSV (one row per record and subset).")
    p.add_argument("--outliers", type=int, default=0,
                   help="Print the K structures with the largest per-structure "
                        "force RMS and the share of the squared force error they "
                        "carry (SPICE diagnostics). Default 0 (off).")
    p.add_argument("--allow_pickle", action="store_true",
                   help="Load checkpoints with torch.load(weights_only=False). "
                        "Only for checkpoints you trust; needed only if a "
                        "checkpoint contains non-tensor Python objects.")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p


# ---------------------------------------------------------------------------
# Job expansion ({molecule} / {seed} templating)
# ---------------------------------------------------------------------------

def _expand(template, molecule, seed):
    out = template
    if molecule is not None:
        out = out.replace("{molecule}", str(molecule))
    if seed is not None:
        out = out.replace("{seed}", str(seed))
    return out


def infer_molecule_and_split(data_path):
    """('aspirin', 'test') from '.../aspirin_test_default.npz'; best effort."""
    stem = os.path.basename(data_path)
    stem = stem[:-4] if stem.endswith(".npz") else stem
    m = re.match(r"^(.*?)_(train|val|test)(?:_.*)?$", stem)
    if m:
        return m.group(1), m.group(2)
    return stem, "unknown"


def expand_jobs(args):
    """Return a list of dicts {molecule, seed, checkpoint, data, config}."""
    ckpts, datas = list(args.checkpoint), list(args.data)
    if len(datas) not in (1, len(ckpts)):
        raise SystemExit("error: give one --data for all checkpoints, or exactly one "
                         "--data per --checkpoint (got %d checkpoints, %d data)"
                         % (len(ckpts), len(datas)))
    if len(datas) == 1:
        datas = datas * len(ckpts)
    templates = ckpts + datas + list(args.config or [])
    uses_mol = any("{molecule}" in t for t in templates)
    uses_seed = any("{seed}" in t for t in templates)
    if uses_mol and not args.molecules:
        raise SystemExit("error: a path contains '{molecule}' but --molecules is not set")
    if uses_seed and not args.seeds:
        raise SystemExit("error: a path contains '{seed}' but --seeds is not set")
    if args.molecules and not uses_mol:
        raise SystemExit("error: --molecules given but no path contains '{molecule}'")
    if args.seeds and not uses_seed:
        raise SystemExit("error: --seeds given but no path contains '{seed}'")

    jobs = []
    for mol in (args.molecules or [None]):
        for seed in (args.seeds or [None]):
            for c, d in zip(ckpts, datas):
                jobs.append({
                    "molecule": mol,
                    "seed": seed,
                    "checkpoint": _expand(c, mol, seed),
                    "data": _expand(d, mol, seed),
                    "config": [_expand(x, mol, seed) for x in (args.config or [])],
                })
    return jobs


# ---------------------------------------------------------------------------
# Config / model / weights (also imported by scripts/utils/*)
# ---------------------------------------------------------------------------

def load_config_files(paths):
    """Merge JSON/YAML config files in order (later files override earlier)."""
    cfg = {}
    for path in paths:
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                part = yaml.safe_load(f) or {}
            else:
                part = json.load(f)
        if not isinstance(part, dict):
            raise ValueError(f"config {path} is not a mapping")
        cfg.update(part)
    return cfg


def resolve_train_yaml(paths, molecule=None):
    """Resolve YAML config files exactly as train.py does (its argparse defaults =
    the paper model, later files override earlier ones)."""
    try:
        import train  # repo-root train.py
        resolver = train.parse_args
    except (ImportError, AttributeError):
        logger.warning("train.parse_args not importable; merging %s WITHOUT train.py's "
                       "defaults (missing keys fall back to legacy model defaults)", paths)
        return load_config_files(paths)
    try:
        argv = ["--dry_run"] + [x for p in paths for x in ("--config", p)]
        if molecule is not None:
            argv += ["--molecule", str(molecule)]   # only fills {molecule} placeholders
        _parser, ns = resolver(argv)
    except SystemExit as exc:
        raise ValueError(f"train.py could not resolve config(s) {paths} (exit {exc.code})")
    cfg = vars(ns).copy()
    for key in ("dry_run", "config"):
        cfg.pop(key, None)
    return cfg


def load_train_config(ckpt_path, config_paths=None, molecule=None):
    """Training config for a checkpoint: --config files, else sibling args.json.

    JSON files (args.json of a run) are used verbatim: keys they lack keep the
    legacy defaults of the code that wrote them.  If any YAML file is given, the
    list is resolved through train.py (paper defaults + overrides), as training
    would; train.py also accepts args.json files in that list.
    """
    if config_paths:
        logger.info("Model config from %s", ", ".join(config_paths))
        if any(p.endswith((".yaml", ".yml")) for p in config_paths):
            return resolve_train_yaml(config_paths, molecule)
        return load_config_files(config_paths)
    args_json = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "args.json")
    if not os.path.exists(args_json):
        raise FileNotFoundError(
            f"no args.json next to {ckpt_path}; pass the training config with --config")
    logger.info("Model config from %s", args_json)
    return load_config_files([args_json])


def resolve_kv_modes(train_cfg):
    """(k_mode, v_mode); old args.json files may carry null next to kv_mode."""
    base = train_cfg.get("kv_mode") or "o2"
    return (train_cfg.get("k_mode") or base, train_cfg.get("v_mode") or base)


# Used for a training config that records no matmul precision (written by an older
# version of this code, which trained with TF32 matmuls).
UNRECORDED_MATMUL_PRECISION = "high"


def resolve_matmul_precision(train_cfg, requested=None):
    """``(precision, source)`` for evaluating a checkpoint trained with ``train_cfg``.

    An explicit ``requested`` value (``--matmul_precision``) always wins.  Else the
    precision train.py recorded for the run: ``_provenance["matmul_precision"]``,
    then the top-level ``matmul_precision`` key (both written by
    train.py as of this release; YAML configs resolved through train.py also carry the
    key).  A config with neither predates the key and gets
    ``UNRECORDED_MATMUL_PRECISION`` ('high', TF32).
    """
    if requested:
        return requested, "explicit"
    prov = train_cfg.get("_provenance")
    if isinstance(prov, dict) and prov.get("matmul_precision"):
        return prov["matmul_precision"], "training config (_provenance)"
    if train_cfg.get("matmul_precision"):
        return train_cfg["matmul_precision"], "training config"
    return UNRECORDED_MATMUL_PRECISION, "not recorded (config predates the key: TF32)"


def make_trainer_config(train_cfg, device, out_dir, matmul_precision=None):
    """Turn a training config (args.json) into a Trainer.from_config dict for
    evaluation: same architecture, no optimizer state / EMA copy / wandb / SWA,
    and nothing is written except an (existing) out_dir.  ``matmul_precision``
    None = the checkpoint's own (:func:`resolve_matmul_precision`)."""
    if matmul_precision is None:
        matmul_precision, _ = resolve_matmul_precision(train_cfg)
    cfg = dict(train_cfg)
    cfg["k_mode"], cfg["v_mode"] = resolve_kv_modes(train_cfg)
    cfg.update({
        "device": device,
        "out_dir": out_dir,
        "_skip_save_args": True,     # never overwrite the training args.json
        "wandb_project": None,
        "wandb_name": None,
        "ema_decay": 0.0,            # weights are loaded directly (see load_weights)
        "swa_start": 0,
        "optimizer": "adamw",        # optimizer is irrelevant for evaluation
        "resume": None,
        "tensor_subclass": False,
        "grad_cos_every": 0,
        "matmul_precision": matmul_precision,
    })
    return cfg


def build_trainer(train_cfg, device, out_dir, matmul_precision=None):
    from AtomTrick.trainer import Trainer
    return Trainer.from_config(make_trainer_config(
        train_cfg, device, out_dir, matmul_precision))


def build_model_from_config(train_cfg):
    """Just the model (CPU, float32) for a training config; no Trainer needed."""
    from AtomTrick.trainer import Trainer
    cfg = dict(train_cfg)
    cfg["k_mode"], cfg["v_mode"] = resolve_kv_modes(train_cfg)
    if hasattr(Trainer, "build_model"):
        return Trainer.build_model(cfg)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        return Trainer.from_config(make_trainer_config(cfg, "cpu", tmp)).model


def read_state_dict(ckpt_path, allow_pickle=False):
    """Return (state_dict, info) for any AtomTrick checkpoint flavour.

    Equivalent to what test.py / eval_mace_metrics.py did with
    ``trainer.load_checkpoint`` + ``trainer._ema_apply()`` for full dicts, and a
    direct ``load_state_dict`` for ``*_best.pt``, without rebuilding optimizer
    or scheduler state.
    """
    import torch
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=not allow_pickle)
    except Exception as exc:  # noqa: BLE001 - re-raised with a hint
        if allow_pickle:
            raise
        raise RuntimeError(
            f"torch.load(weights_only=True) failed for {ckpt_path}: {exc}\n"
            "If you trust this file, re-run with --allow_pickle.") from exc
    info = {"step": None}
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        info["step"] = obj.get("step")
        ema = obj.get("ema_model")
        if ema:
            # AveragedModel.state_dict() = {"module.<name>": ..., "n_averaged": ...}
            sd = {k[len("module."):]: v for k, v in ema.items() if k.startswith("module.")}
            info["weights"] = "ema (full checkpoint)"
        else:
            sd = obj["model"]
            info["weights"] = "raw training weights (full checkpoint without EMA)"
            logger.warning("%s has no EMA state; evaluating the raw training weights",
                           ckpt_path)
    elif isinstance(obj, dict):
        sd = obj
        info["weights"] = "state_dict (EMA snapshot, *_best.pt)"
    else:
        raise ValueError(f"unrecognised checkpoint format in {ckpt_path}: {type(obj)}")
    return sd, info


def load_weights(model, state_dict):
    """Load weights with strict=True semantics for PARAMETERS.

    Unexpected keys and missing parameters are errors (as with strict=True).
    Missing *buffers* are accepted: they are non-learned constants that the
    constructor rebuilds (e.g. index tables absent from stripped checkpoints).
    Legacy key layouts are converted by the model's load_state_dict pre-hooks.
    """
    res = model.load_state_dict(state_dict, strict=False)
    params = {n for n, _ in model.named_parameters()}
    missing_params = [k for k in res.missing_keys if k in params]
    if res.unexpected_keys or missing_params:
        raise RuntimeError(
            "checkpoint does not match the model built from the config:\n"
            f"  missing parameters: {missing_params[:10]}{' ...' if len(missing_params) > 10 else ''}\n"
            f"  unexpected keys:    {res.unexpected_keys[:10]}"
            f"{' ...' if len(res.unexpected_keys) > 10 else ''}\n"
            "Is the config (args.json / --config) the one this checkpoint was trained with?")
    if res.missing_keys:
        logger.info("%d non-learned buffers not in checkpoint; using constructed values",
                    len(res.missing_keys))


def check_atomic_refs(model, atomic_ref):
    """Warn if the dataset's reference energies differ from the model buffer.

    Energy MAEs are only meaningful when the labels are referenced with the
    same per-element table the model was trained with (args.json 'atomic_ref',
    or the default table when it is null).
    """
    import torch
    from AtomTrick.constants import ATOMIC_REFERENCE_ENERGIES
    table = atomic_ref if atomic_ref is not None else ATOMIC_REFERENCE_ENERGIES
    buf = getattr(model, "atomic_ref_energy", None)
    if buf is None:
        return
    buf = buf.detach().float().cpu()
    bad = []
    for z, e in table.items():
        z = int(z)
        if z < buf.numel():
            ref32 = torch.tensor(float(e), dtype=torch.float32).item()
            if abs(buf[z].item() - ref32) > 1e-6 * abs(ref32) + 1e-6:
                bad.append((z, float(e), buf[z].item()))
    if bad:
        logger.warning("atomic reference energies in the config differ from the "
                       "checkpoint's atomic_ref_energy buffer for Z=%s; energy "
                       "errors may be offset. (z, config, checkpoint): %s",
                       [b[0] for b in bad], bad[:5])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataset(path, train_cfg):
    import torch
    from AtomTrick.data import MLFFDataset
    dtype = torch.float64 if train_cfg.get("dtype") == "float64" else torch.float32
    return MLFFDataset(
        path,
        edge_cutoff=train_cfg.get("edge_cutoff", 5.0),
        dtype=dtype,
        atomic_ref=train_cfg.get("atomic_ref"),
        max_neighbors=train_cfg.get("max_neighbors"),
    )


def load_subsets(path, n_keep):
    """Per-structure canonical subset names, or None if the NPZ has none."""
    with np.load(path, allow_pickle=True) as d:
        if "subset" not in d:
            return None
        ids = np.asarray(d["subset"]).astype(np.int64)[:n_keep]
        if "subset_names" in d:
            names = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                     for x in list(d["subset_names"])]
        else:
            names = [f"subset_{i}" for i in range(int(ids.max()) + 1)]
    lookup = np.array([canonical_subset_name(n) for n in names], dtype=object)
    return lookup[ids]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pooled_metrics(per, mask=None):
    """Pooled metrics (meV, meV/Å) from per-structure float64 sums."""
    if mask is None:
        mask = np.ones(len(per["e_err"]), dtype=bool)
    e = per["e_err"][mask]
    nat = per["n_atoms"][mask]
    n_struct = int(mask.sum())
    n_atoms = int(nat.sum())
    if n_struct == 0:
        return {"n_structures": 0, "n_atoms": 0}
    ncomp = 3.0 * n_atoms
    ea = e / nat
    return {
        "e_mae_meV": float(np.abs(e).mean() * 1e3),
        "e_rmse_meV": float(math.sqrt(np.mean(e ** 2)) * 1e3),
        "e_mae_meV_atom": float(np.abs(ea).mean() * 1e3),
        "e_rmse_meV_atom": float(math.sqrt(np.mean(ea ** 2)) * 1e3),
        "f_mae_meVA": float(per["f_abs"][mask].sum() / ncomp * 1e3),
        "f_rmse_meVA": float(math.sqrt(per["f_sq"][mask].sum() / ncomp) * 1e3),
        "f_l2_meVA": float(per["f_l2"][mask].sum() / n_atoms * 1e3),
        "n_structures": n_struct,
        "n_atoms": n_atoms,
    }


def _forward_fn(trainer):
    fwd = getattr(trainer, "_forward", None)
    return fwd if fwd is not None else trainer.model


def run_inference(trainer, dataset, eval_bs, num_workers=0):
    """One pass over ``dataset``; returns per-structure float64 error sums."""
    import torch
    from AtomTrick.data import build_dataloader

    loader = build_dataloader(dataset, {"batch_size": eval_bs, "num_workers": num_workers},
                              train=False)
    model = trainer.model
    model.eval()
    dev = trainer.device
    forward = _forward_fn(trainer)
    f64 = torch.float64
    cols = {k: [] for k in ("e_err", "n_atoms", "f_abs", "f_sq", "f_l2")}
    batch_fmae = []
    n_done, n_total = 0, len(dataset)
    t0 = t_last = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            pred = forward(batch)
            n_mol = int(batch["N"].shape[0])
            mol = batch["batch"]
            bias = model.energy_bias_per_mol(batch["Z"], mol, n_mol)
            e_err = pred["E"].to(f64) + bias.to(f64) - batch["E"].to(f64)   # eV
            f_err = pred["F"].to(f64) - batch["F"].to(f64)                   # eV/Å
            zeros = torch.zeros(n_mol, dtype=f64, device=dev)
            cols["e_err"].append(e_err)
            cols["n_atoms"].append(batch["N"].to(f64))
            cols["f_abs"].append(zeros.index_add(0, mol, f_err.abs().sum(-1)))
            cols["f_sq"].append(zeros.index_add(0, mol, (f_err ** 2).sum(-1)))
            cols["f_l2"].append(zeros.index_add(0, mol, f_err.norm(dim=-1)))
            batch_fmae.append(f_err.abs().mean())
            n_done += n_mol
            if time.time() - t_last > 60:
                t_last = time.time()
                logger.info("  %d / %d structures (%.1f struct/s)", n_done, n_total,
                            n_done / (t_last - t0))
    per = {k: torch.cat(v).cpu().numpy() for k, v in cols.items()}
    per["batch_fmae"] = float(torch.stack(batch_fmae).mean().item()) if batch_fmae else float("nan")
    per["seconds"] = time.time() - t0
    return per


def run_inference_with_oom_retry(trainer, dataset, eval_bs, num_workers, makefx):
    """Eager mode: on CUDA OOM, halve the batch size and restart the pass."""
    import torch
    bs = eval_bs
    while True:
        try:
            return run_inference(trainer, dataset, bs, num_workers), bs
        except torch.cuda.OutOfMemoryError:
            if makefx or bs == 1:
                raise
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            logger.warning("CUDA out of memory; retrying with --eval_bs %d", bs)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - git missing / not a checkout
        return None


# ---------------------------------------------------------------------------
# One evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(ckpt_path, data_path, *, config_paths=None, device=None,
                        eval_bs=8, per_subset=False,
                        limit=None, num_workers=0, makefx=False,
                        matmul_precision=None, allow_pickle=False, tag=None,
                        molecule=None, seed=None, outliers=0, escn_frame=None):
    """Evaluate one checkpoint on one NPZ and return a result record (dict).

    ``matmul_precision`` None = the precision the checkpoint was trained with
    (:func:`resolve_matmul_precision`; 'high' = TF32 when it records none)."""
    import torch
    from torch.utils.data import Subset

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data not found: {data_path}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cfg = load_train_config(ckpt_path, config_paths, molecule)
    if escn_frame is not None:
        train_cfg = dict(train_cfg, escn_frame=escn_frame)
    matmul_precision, precision_source = resolve_matmul_precision(train_cfg, matmul_precision)
    if (precision_source == "training config" and config_paths
            and any(p.endswith((".yaml", ".yml")) for p in config_paths)):
        # train.py's resolution always fills the key (YAML value, else its default):
        # not something the run recorded.
        precision_source = "YAML --config (resolved through train.py)"
    if precision_source.startswith("not recorded"):
        logger.info("The training config records no matmul precision (it predates "
                    "the key; such runs used TF32): evaluating with "
                    "matmul precision '%s'. Pass --matmul_precision to override.",
                    matmul_precision)
    else:
        logger.info("Matmul precision: %s (%s)", matmul_precision, precision_source)

    out_dir = os.path.dirname(os.path.abspath(ckpt_path))
    trainer = build_trainer(train_cfg, device, out_dir, matmul_precision)
    # Trainer.from_config already set it for fp32 models; set it for every dtype.
    torch.set_float32_matmul_precision(matmul_precision)
    sd, winfo = read_state_dict(ckpt_path, allow_pickle=allow_pickle)
    load_weights(trainer.model, sd)
    del sd
    trainer.model.eval()
    if train_cfg.get("atomic_ref") is None and train_cfg.get("fit_atomic_refs"):
        # A YAML config cannot know the reference energies fitted at training
        # time; recover them from the checkpoint buffer (float32-rounded).
        buf = trainer.model.atomic_ref_energy.detach().double().cpu()
        train_cfg = dict(train_cfg, atomic_ref={
            z: float(buf[z]) for z in range(buf.numel()) if buf[z] != 0})
        logger.warning("config has fit_atomic_refs but no 'atomic_ref' table: using the "
                       "checkpoint's float32 atomic_ref_energy buffer. Energies can be off "
                       "by ~1e-3 eV/atom for heavy elements; prefer the run's args.json.")
    check_atomic_refs(trainer.model, train_cfg.get("atomic_ref"))

    dataset = build_dataset(data_path, train_cfg)
    n_all = len(dataset)
    if limit is not None and limit < n_all:
        dataset = Subset(dataset, range(max(0, limit)))
    n_eval = len(dataset)
    if n_eval == 0:
        raise ValueError(f"no structures to evaluate in {data_path}")
    logger.info("Evaluating %d / %d structures of %s (eval_bs %d, float64 labels, %s)",
                n_eval, n_all, data_path, eval_bs, "make_fx" if makefx else "eager")

    per, used_bs = run_inference_with_oom_retry(trainer, dataset, eval_bs, num_workers, makefx)
    metrics = pooled_metrics(per)

    inferred_mol, split = infer_molecule_and_split(data_path)
    seed_val = seed if seed is not None else train_cfg.get("seed")
    try:
        seed_val = int(seed_val) if seed_val is not None else None
    except (TypeError, ValueError):
        pass
    if seed is not None and train_cfg.get("seed") is not None \
            and str(train_cfg.get("seed")) != str(seed):
        logger.warning("--seeds value %s differs from the config seed %s",
                       seed, train_cfg.get("seed"))
    exp = train_cfg.get("experiment")
    record = {
        "tag": tag or (exp if exp and exp != "default" else "default"),
        "molecule": molecule or inferred_mol,
        "split": split,
        "seed": seed_val,
        "checkpoint": ckpt_path,
        "data": data_path,
        "label_mode": "float64",
        "escn_frame": trainer.model.escn_frame,
        "weights": winfo["weights"],
        "step": winfo.get("step"),
        **metrics,
        "f_mae_batchavg_meVA": per["batch_fmae"] * 1e3,
        "eval": {
            "eval_bs": used_bs,
            "makefx": bool(makefx),
            "device": str(trainer.device),
            "device_name": (torch.cuda.get_device_name(trainer.device)
                            if trainer.device.type == "cuda" else "cpu"),
            "matmul_precision": matmul_precision,
            "matmul_precision_source": precision_source,
            "limit": limit,
            "n_structures_in_file": n_all,
            "seconds": round(per["seconds"], 2),
        },
        "model": {"n_params": count_parameters(trainer.model),
                  **{k: train_cfg.get(k) for k in MODEL_KEYS if k in train_cfg}},
        "provenance": {
            "torch": torch.__version__,
            "git_sha": git_sha(),
            "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    }
    # Record the K/V modes the model was actually built with.
    record["model"]["k_mode"], record["model"]["v_mode"] = resolve_kv_modes(train_cfg)

    if per_subset:
        names = load_subsets(data_path, n_eval)
        if names is None:
            logger.warning("--per_subset: %s has no 'subset' array; skipped", data_path)
        else:
            order = [s for s in SPICE_SUBSET_ORDER if s in set(names)]
            order += sorted(set(names) - set(order))
            record["per_subset"] = {s: pooled_metrics(per, names == s) for s in order}

    if outliers:
        record["outliers"] = force_outlier_summary(per, outliers, data_path, n_eval)

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def force_outlier_summary(per, k, data_path, n_eval):
    """Worst-K structures by per-structure force RMS + share of squared error."""
    f_rms = np.sqrt(per["f_sq"] / (3.0 * per["n_atoms"])) * 1e3
    order = np.argsort(-f_rms)
    names = load_subsets(data_path, n_eval)
    tot = per["f_sq"].sum()
    shares = {}
    for frac in (0.001, 0.01, 0.05):
        kk = max(1, int(len(f_rms) * frac))
        shares[f"worst_{frac * 100:g}pct"] = float(per["f_sq"][order[:kk]].sum() / tot * 100)
    worst = [{"index": int(i), "f_rms_meVA": float(f_rms[i]),
              "n_atoms": int(per["n_atoms"][i]),
              "subset": (str(names[i]) if names is not None else None)}
             for i in order[:k]]
    return {"share_of_squared_force_error_pct": shares,
            "f_rms_percentiles_meVA": {p: float(np.percentile(f_rms, p))
                                       for p in (50, 90, 99, 99.9)},
            "worst": worst}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(v, nd):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def print_records(records, file=sys.stdout):
    if not records:
        return
    hdr = (f"{'tag':<14} {'molecule':<24} {'split':<6} {'N':>7} "
           f"{'E_MAE':>9} {'E_RMSE':>9} {'E_MAE/at':>9} {'E_RMSE/at':>9} "
           f"{'F_MAE':>8} {'F_RMSE':>8} {'F_L2':>8}")
    units = (f"{'':<14} {'':<24} {'':<6} {'':>7} "
             f"{'(meV)':>9} {'(meV)':>9} {'(meV/at)':>9} {'(meV/at)':>9} "
             f"{'(meV/A)':>8} {'(meV/A)':>8} {'(meV/A)':>8}")
    print(hdr, file=file)
    print(units, file=file)
    print("-" * len(hdr), file=file)
    for r in records:
        print(f"{str(r['tag'])[:14]:<14} {str(r['molecule'])[:24]:<24} {r['split']:<6} "
              f"{r['n_structures']:>7d} "
              f"{_fmt(r['e_mae_meV'], 3):>9} {_fmt(r['e_rmse_meV'], 3):>9} "
              f"{_fmt(r['e_mae_meV_atom'], 4):>9} {_fmt(r['e_rmse_meV_atom'], 4):>9} "
              f"{_fmt(r['f_mae_meVA'], 3):>8} {_fmt(r['f_rmse_meVA'], 3):>8} "
              f"{_fmt(r['f_l2_meVA'], 3):>8}", file=file)
    print("-" * len(hdr), file=file)
    for r in records:
        if r.get("per_subset"):
            print(f"\nPer subset: {r['tag']} / {r['molecule']} ({r['checkpoint']})", file=file)
            sh = (f"{'subset':<22} {'N':>7} {'E_MAE/at':>9} {'E_RMSE/at':>9} "
                  f"{'F_MAE':>8} {'F_RMSE':>8} {'F_L2':>8}")
            print(sh, file=file)
            print("-" * len(sh), file=file)
            rows = list(r["per_subset"].items()) + [("OVERALL", r)]
            for name, m in rows:
                if not m.get("n_structures"):
                    continue
                print(f"{name[:22]:<22} {m['n_structures']:>7d} "
                      f"{_fmt(m['e_mae_meV_atom'], 3):>9} {_fmt(m['e_rmse_meV_atom'], 3):>9} "
                      f"{_fmt(m['f_mae_meVA'], 3):>8} {_fmt(m['f_rmse_meVA'], 3):>8} "
                      f"{_fmt(m['f_l2_meVA'], 3):>8}", file=file)
        if r.get("outliers"):
            o = r["outliers"]
            print(f"\nForce outliers ({r['molecule']}): share of squared force error "
                  + ", ".join(f"{k} {v:.1f}%" for k, v in
                              o["share_of_squared_force_error_pct"].items()), file=file)
            for w in o["worst"]:
                print(f"  f_rms {w['f_rms_meVA']:9.1f} meV/A  index {w['index']:>7d}  "
                      f"atoms {w['n_atoms']:>4d}  {w['subset'] or ''}", file=file)


def write_json(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, path)


def flatten_rows(records):
    """One CSV row per record (subset 'ALL') plus one per subset."""
    rows = []
    for r in records:
        base = {k: v for k, v in r.items()
                if k not in ("per_subset", "eval", "model", "provenance", "outliers")}
        base.update({f"eval_{k}": v for k, v in r.get("eval", {}).items()})
        base.update({f"model_{k}": v for k, v in r.get("model", {}).items()})
        base.update({f"prov_{k}": v for k, v in r.get("provenance", {}).items()})
        rows.append({"subset": "ALL", **base})
        for name, m in (r.get("per_subset") or {}).items():
            row = {k: v for k, v in base.items() if k not in METRIC_KEYS
                   and k not in ("n_structures", "n_atoms", "f_mae_batchavg_meVA")}
            row.update({"subset": name, **m})
            rows.append(row)
    return rows


def write_csv(records, path):
    rows = flatten_rows(records)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def configure_environment(makefx):
    """Must run BEFORE torch / AtomTrick are imported."""
    os.environ["ATOMTRICK_MAKEFX"] = "1" if makefx else "0"
    os.environ.setdefault("PYTORCH_JIT", "0")   # make_fx needs it; harmless in eager
    stale = sorted(k for k in os.environ if k.startswith("ATOMTRICKS_"))
    if stale:
        logger.warning("Ignoring old-spelling environment variables %s (the package "
                       "was renamed AtomTricks -> AtomTrick; use ATOMTRICK_*).", stale)


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    if args.eval_bs < 1:
        raise SystemExit("error: --eval_bs must be >= 1")
    jobs = expand_jobs(args)
    makefx = args.makefx == "on"
    if makefx:
        # Disable make_fx duck-shaping when every evaluated structure has the same
        # atom count (rMD17 / MD22), exactly as train.py does for training.
        sizes = set()
        for j in jobs:
            if os.path.exists(j["data"]):
                with np.load(j["data"]) as d:
                    sizes.update(np.unique(np.asarray(d["N"])).tolist())
        os.environ["ATOMTRICK_MAKEFX_DUCKOFF"] = "1" if len(sizes) == 1 else "0"
    configure_environment(makefx)

    records, failures = [], []
    for i, job in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {job['checkpoint']}\n        on {job['data']}", flush=True)
        try:
            rec = evaluate_checkpoint(
                job["checkpoint"], job["data"], config_paths=job["config"] or None,
                device=args.device, eval_bs=args.eval_bs,
                per_subset=args.per_subset,
                limit=args.limit, num_workers=args.num_workers, makefx=makefx,
                matmul_precision=args.matmul_precision, allow_pickle=args.allow_pickle,
                tag=args.tag, molecule=job["molecule"], seed=job["seed"],
                outliers=args.outliers, escn_frame=args.escn_frame)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            logger.error("FAILED %s: %s", job["checkpoint"], exc)
            logger.debug("%s", traceback.format_exc())
            failures.append((job["checkpoint"], str(exc)))
            continue
        records.append(rec)
        print_records([rec])
        if args.out:
            write_json(records, args.out)

    if len(records) > 1:
        print("\nSummary")
        print_records(records)
    if args.out:
        write_json(records, args.out)
        print(f"\nWrote {len(records)} record(s) to {args.out}")
    if args.csv:
        write_csv(records, args.csv)
        print(f"Wrote CSV to {args.csv}")
    if failures:
        print(f"\n{len(failures)} evaluation(s) FAILED:", file=sys.stderr)
        for c, e in failures:
            print(f"  {c}: {e.splitlines()[0] if e else ''}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
