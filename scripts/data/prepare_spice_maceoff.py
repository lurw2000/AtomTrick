#!/usr/bin/env python3
"""Convert the MACE-OFF23 SPICE release (extended xyz) into ATOMTRICK NPZs.

Builds, in one tool, the SPICE data files of the paper -- converter, force
statistics, filtered test:

1. **Labels.**  The DFT reference is the ``energy=`` key of each comment line
   and the ``forces`` per-atom columns.  ``MACE_energy``/``MACE_forces`` are
   MACE-OFF23's own predictions and are ignored.  Units are converted eV ->
   kcal/mol (``1 / (ase.units.kcal / ase.units.mol)``) because the training
   loader converts kcal/mol back to eV.  E is stored as float64, R and F as
   float32, Z as int32.  ``subset`` (int per structure) and ``subset_names``
   (sorted distinct ``config_type`` strings of that file, a pickled object
   array) record the SPICE subset.
2. **Train/val split** of ``train_large_neut_no_bad_clean.xyz``: a
   structure-level random split, ``np.random.default_rng(42).permutation``,
   ``round(0.05 * M)`` structures to validation, both sides kept in file order.
3. **Test** = ``test_large_neut_all.xyz`` converted unchanged
   (``spice_maceoff_test.npz``): the input of the filter below.  It is not
   evaluated by the release scripts.
4. **Filtered test (optional, on by default)**: the force thresholds MACE-OFF applies
   when cleaning its SPICE 2 training data, applied here to the released test file:
   keep a structure iff max_atom ||F_atom|| <= 15 eV/A and
   ||sum_atoms F|| <= 0.1 eV/A (both from the DFT labels, in float64).  This file (``spice_maceoff_test_f15.npz``) is
   the SPICE test set: all reported SPICE results are computed on it only.

Expected counts with the defaults:
    all 951,005 -> train 903,455 / val 47,550; test 50,195; test_f15 50,069
    (126 PubChem structures dropped: 121 by max-force, 5 by net force)

Usage (from the repo root, after scripts/data/download_spice_maceoff.sh)::

    python scripts/data/prepare_spice_maceoff.py --raw_dir data/raw/spice_maceoff --out_dir data/spice_maceoff
    python scripts/data/verify_data.py --only data/spice_maceoff

Parsing the 5 GB training xyz is pure Python and takes a while (tens of
minutes) and several GB of RAM; ``--skip_train`` builds only the test files.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase import units
from ase.data import atomic_numbers

EV2KCAL = 1.0 / (units.kcal / units.mol)     # ~23.0605; inverse of the loader's conversion
KCAL2EV = units.kcal / units.mol

_energy_re = re.compile(r'(?:^|\s)energy=(-?\d+\.?\d*(?:[eE][+-]?\d+)?)')
_ct_re = re.compile(r'config_type=("([^"]*)"|(\S+))')

TRAIN_XYZ = "train_large_neut_no_bad_clean.xyz"
TEST_XYZ = "test_large_neut_all.xyz"
EXPECTED = {"all": 951005, "train": 903455, "val": 47550, "test": 50195, "test_f15": 50069}


# --------------------------------------------------------------------------- parsing
def _col_offsets(comment):
    """Parse ``Properties=...`` to find the start column of each per-atom field."""
    m = re.search(r'Properties=(\S+)', comment)
    fields = m.group(1).split(':')
    off, col, i = {}, 0, 0
    while i < len(fields):
        name, _typ, width = fields[i], fields[i + 1], int(fields[i + 2])
        off[name] = (col, width)
        col += width
        i += 3
    return off, col


def parse_xyz(path):
    """Read a MACE-OFF23 extended-xyz file into flat arrays (kcal/mol, A)."""
    N, Zc, Rc, Fc, Ec, subc = [], [], [], [], [], []
    with open(path) as f:
        first = True
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line == '':
                continue
            nat = int(line)
            comment = f.readline()
            if first:
                off, _ = _col_offsets(comment)
                sp0, ps0, fr0 = off['species'][0], off['pos'][0], off['forces'][0]
                first = False
            E = float(_energy_re.search(comment).group(1))
            mct = _ct_re.search(comment)
            ct = (mct.group(2) if mct.group(2) is not None else mct.group(3)) if mct else 'UNKNOWN'
            Z = np.empty(nat, dtype=np.int32)
            R = np.empty((nat, 3), dtype=np.float32)
            Fo = np.empty((nat, 3), dtype=np.float32)
            for a in range(nat):
                t = f.readline().split()
                Z[a] = atomic_numbers[t[sp0]]
                R[a] = (float(t[ps0]), float(t[ps0 + 1]), float(t[ps0 + 2]))
                Fo[a] = (float(t[fr0]) * EV2KCAL, float(t[fr0 + 1]) * EV2KCAL,
                         float(t[fr0 + 2]) * EV2KCAL)
            N.append(nat)
            Zc.append(Z)
            Rc.append(R)
            Fc.append(Fo)
            Ec.append(E * EV2KCAL)
            subc.append(ct)
    N = np.asarray(N, dtype=np.int64)
    Z = np.concatenate(Zc)
    R = np.concatenate(Rc)
    F = np.concatenate(Fc)
    E = np.asarray(Ec, dtype=np.float64)
    names = sorted(set(subc))
    name2id = {nm: i for i, nm in enumerate(names)}
    subset = np.asarray([name2id[s] for s in subc], dtype=np.int64)
    return dict(N=N, Z=Z, R=R, F=F, E=E, subset=subset,
                subset_names=np.array(names, dtype=object))


# --------------------------------------------------------------------------- helpers
def select_structures(d, keep):
    """Rows of ``d`` for the structures where boolean ``keep`` is True (file order kept)."""
    struct_of_atom = np.repeat(np.arange(len(d["N"])), d["N"])
    amask = keep[struct_of_atom]
    return dict(N=d["N"][keep], Z=d["Z"][amask], R=d["R"][amask], F=d["F"][amask],
                E=d["E"][keep], subset=d["subset"][keep], subset_names=d["subset_names"])


def report(tag, d):
    names = [str(x) for x in d["subset_names"]]
    print(f"[{tag}] structs={len(d['N']):,} atoms={len(d['Z']):,} "
          f"elems={sorted(set(np.unique(d['Z']).tolist()))}")
    for sid, cnt in sorted(Counter(d["subset"].tolist()).items(), key=lambda x: -x[1]):
        print(f"    {cnt:>8,}  {names[sid]}")


def check_count(tag, n, published):
    if published and tag in EXPECTED and n != EXPECTED[tag]:
        print(f"    WARNING: {tag} has {n:,} structures, expected {EXPECTED[tag]:,}")
        return False
    return True


def force_stats(d):
    """Per-structure max atomic force norm and net-force norm, in eV/A.

    Same arithmetic as the original research script (float64, per-structure loop),
    so threshold decisions are reproduced exactly.
    """
    F = d["F"].astype(np.float64).reshape(-1, 3) * KCAL2EV
    off = np.concatenate([[0], np.cumsum(d["N"])])
    M = len(d["N"])
    fmax = np.zeros(M)
    fnet = np.zeros(M)
    for i in range(M):
        f = F[off[i]:off[i + 1]]
        fmax[i] = np.linalg.norm(f, axis=1).max()
        fnet[i] = np.linalg.norm(f.sum(0))
    return fmax, fnet


def save(path, d):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    print(f"WROTE {path}")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw_dir", default="data/raw/spice_maceoff",
                    help=f"directory with {TRAIN_XYZ} and {TEST_XYZ}")
    ap.add_argument("--out_dir", default="data/spice_maceoff")
    ap.add_argument("--val_frac", type=float, default=0.05, help="validation fraction (paper: 0.05)")
    ap.add_argument("--seed", type=int, default=42, help="train/val split seed (paper: 42)")
    ap.add_argument("--fmax", type=float, default=15.0,
                    help="filtered test: max per-atom |F| in eV/A (MACE-OFF23: 15)")
    ap.add_argument("--fnet", type=float, default=0.1,
                    help="filtered test: max |sum F| in eV/A (MACE-OFF23: 0.1)")
    ap.add_argument("--no_filtered_test", action="store_true",
                    help="do not write spice_maceoff_test_f15.npz")
    ap.add_argument("--skip_train", action="store_true", help="only build the test file(s)")
    ap.add_argument("--skip_test", action="store_true", help="only build train/val")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out_dir)
    published = (args.val_frac, args.seed, args.fmax, args.fnet) == (0.05, 42, 15.0, 0.1)
    ok = True
    t0 = time.time()

    # ---- test (+ filtered test) -------------------------------------------------
    if not args.skip_test:
        p_test = out / "spice_maceoff_test.npz"
        p_f15 = out / "spice_maceoff_test_f15.npz"
        need_f15 = not args.no_filtered_test and (args.force or not p_f15.exists())
        if args.force or not p_test.exists():
            d = parse_xyz(raw / TEST_XYZ)
            report("TEST", d)
            save(p_test, d)
        elif need_f15:
            print(f"[skip] {p_test} exists; loading it to build the filtered test")
            with np.load(p_test, allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
        else:
            print(f"[skip] {p_test} and {p_f15} exist (use --force to rebuild)")
            d = None
        if d is not None:
            ok &= check_count("test", len(d["N"]), published)
        if d is not None and need_f15:
            fmax, fnet = force_stats(d)
            keep = (fmax <= args.fmax) & (fnet <= args.fnet)
            print(f"[FILTER] keep = (max|F_atom| <= {args.fmax} eV/A) & (|sum F| <= {args.fnet} eV/A): "
                  f"{int(keep.sum()):,} kept, {int((~keep).sum())} dropped "
                  f"({int((fmax > args.fmax).sum())} by max force, {int((fnet > args.fnet).sum())} by net force)")
            names = [str(x) for x in d["subset_names"]]
            for sid, nm in enumerate(names):
                m = d["subset"] == sid
                if m.any():
                    print(f"    {nm:<22} {int(m.sum()):>7,} -> {int((m & keep).sum()):>7,}")
            f15 = select_structures(d, keep)
            save(p_f15, f15)
            ok &= check_count("test_f15", len(f15["N"]), published)

    # ---- train / val ---------------------------------------------------------------
    if not args.skip_train:
        p_tr, p_val = out / "spice_maceoff_train.npz", out / "spice_maceoff_val.npz"
        if p_tr.exists() and p_val.exists() and not args.force:
            print(f"[skip] {p_tr} and {p_val} exist (use --force to rebuild)")
        else:
            print(f"parsing {raw / TRAIN_XYZ} (slow: pure-Python reader) ...")
            d = parse_xyz(raw / TRAIN_XYZ)
            report("ALL", d)
            ok &= check_count("all", len(d["N"]), published)
            M = len(d["N"])
            perm = np.random.default_rng(args.seed).permutation(M)
            nval = int(round(M * args.val_frac))
            is_val = np.zeros(M, dtype=bool)
            is_val[perm[:nval]] = True          # == np.sort(perm[:nval]) in the original script
            dtr, dval = select_structures(d, ~is_val), select_structures(d, is_val)
            del d
            report("TRAIN", dtr)
            report("VAL", dval)
            ok &= check_count("train", len(dtr["N"]), published)
            ok &= check_count("val", len(dval["N"]), published)
            save(p_tr, dtr)
            save(p_val, dval)

    print(f"done in {time.time() - t0:.0f}s; check: python scripts/data/verify_data.py --only data/spice_maceoff")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
