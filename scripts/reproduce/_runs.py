#!/usr/bin/env python3
"""List every training run under a runs/ directory and how to evaluate it.

Helper for scripts/reproduce/eval_all.sh (standard library only).  A run is a
directory containing ``args.json`` (written by train.py).  For each run and each
test set it prints one tab-separated line::

    checkpoint  data  molecule  eval_bs  out_json  tag  per_subset(0/1)

``checkpoint`` / ``data`` contain ``{molecule}`` when ``molecule`` is not ``-``
(evaluate.py expands it over ``--molecules``); ``-`` = no molecule (SPICE).  The test set is derived from the
run's training file: ``<mol>_train_default.npz`` -> ``<mol>_test_default.npz``;
SPICE ``spice_maceoff_train.npz`` -> ``spice_maceoff_test_f15.npz`` (the force-filtered
test set, the only one evaluated).  The checkpoint is ``<wandb_name>_best.pt``, else
the only ``*_best.pt`` in the directory.  Runs without one are reported on stderr and
skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

MD22_EVAL_BS = {"Ac-Ala3-NHMe": 8, "DHA": 8, "AT-AT": 8, "stachyose": 4, "AT-AT-CG-CG": 4,
                "buckyball-catcher": 2, "double-walled_nanotube": 1}
SPICE_EVAL_BS = 16
DEFAULT_EVAL_BS = 8


def find_runs(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ("wandb", "__pycache__")
                             and not d.startswith("."))
        if "args.json" in filenames:
            yield dirpath


def templated(path, mol):
    """Replace the molecule in path components / file stem by '{molecule}'."""
    parts = path.split(os.sep)
    parts = ["{molecule}" if p == mol else p for p in parts]
    parts[-1] = parts[-1].replace(f"_{mol}_", "_{molecule}_")
    if parts[-1].startswith(f"{mol}_"):
        parts[-1] = "{molecule}" + parts[-1][len(mol):]
    return os.sep.join(parts)


def describe(run_dir):
    with open(os.path.join(run_dir, "args.json")) as f:
        args = json.load(f)
    name = args.get("wandb_name")
    candidates = [f"{name}_best.pt" if name else None]
    ckpt = next((os.path.join(run_dir, c) for c in candidates
                 if c and os.path.exists(os.path.join(run_dir, c))), None)
    if ckpt is None:
        found = sorted(glob.glob(os.path.join(run_dir, "*_best.pt")))
        if len(found) > 1:
            print(f"[runs] several *_best.pt in {run_dir}; using {found[0]}", file=sys.stderr)
        ckpt = found[0] if found else None
    if ckpt is None:
        print(f"[runs] no *_best.pt in {run_dir}; skipped", file=sys.stderr)
        return []
    data = args.get("data") or ""
    ddir, base = os.path.split(data)
    tag = args.get("experiment")
    if not tag:
        # Older args.json files have no 'experiment': take it from the layout
        # <experiment>/<molecule>/s<seed>/ (rMD17, MD22) or <experiment>/s<seed>/ (SPICE).
        mol_guess = args.get("molecule") or (base[: -len("_train_default.npz")]
                                             if base.endswith("_train_default.npz") else None)
        parent = os.path.dirname(os.path.abspath(run_dir))
        if mol_guess and os.path.basename(parent) == mol_guess:
            parent = os.path.dirname(parent)
        tag = os.path.basename(parent)
    lines = []
    if base.startswith("spice_maceoff_train"):
        lines.append((ckpt, os.path.join(ddir, "spice_maceoff_test_f15.npz"), "", SPICE_EVAL_BS,
                      os.path.join(run_dir, "eval_test_f15.json"), tag, 1))
    elif base.endswith("_train_default.npz"):
        mol = args.get("molecule") or base[: -len("_train_default.npz")]
        test = os.path.join(ddir, f"{mol}_test_default.npz")
        bs = MD22_EVAL_BS.get(mol, DEFAULT_EVAL_BS)
        c_t, d_t = templated(ckpt, mol), templated(test, mol)
        if "{molecule}" in c_t and "{molecule}" in d_t:
            lines.append((c_t, d_t, mol, bs, os.path.join(run_dir, "eval_test.json"), tag, 0))
        else:
            lines.append((ckpt, test, "", bs, os.path.join(run_dir, "eval_test.json"), tag, 0))
    elif "_train" in base:
        test = os.path.join(ddir, base.replace("_train", "_test"))
        lines.append((ckpt, test, "", DEFAULT_EVAL_BS, os.path.join(run_dir, "eval_test.json"), tag, 0))
    else:
        print(f"[runs] cannot derive a test set from data={data!r} in {run_dir}; skipped",
              file=sys.stderr)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default="runs", help="directory to scan (default: runs)")
    ap.add_argument("--only", default=None, help="only runs whose path contains this substring")
    args = ap.parse_args()
    if not os.path.isdir(args.runs):
        print(f"[runs] no directory {args.runs}", file=sys.stderr)
        return 0
    for run_dir in find_runs(args.runs):
        if args.only and args.only not in run_dir:
            continue
        for rec in describe(run_dir):
            print("\t".join(str(x) if x != "" else "-" for x in rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
