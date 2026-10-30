#!/usr/bin/env python3
"""Build the MD22 train/val/test NPZs used by every ATOMTRICK MD22 run.

Thin CLI over :class:`AtomTrick.data.md22.MD22Processor` with the settings of the
reported runs:

* MD22 has no official split files.  Each system's frames are shuffled with
  ``np.random.RandomState(1)`` (``--seed 1``); the first ``train`` frames are
  the training set, the next ``val`` frames the validation set, and ALL
  remaining frames the test set.
* ``train``/``val`` sizes come from ``DEFAULT_SIZES`` in ``AtomTrick/data/md22.py``.
  The training-set sizes are those of Chmiela et al. (2023); the validation
  sizes (500, or 100 for the two largest systems) and the random split are our
  choice.

    system                  atoms  train  val   test
    Ac-Ala3-NHMe               42   6000  500  78609
    DHA                        56   8000  500  61253
    AT-AT                      60   3000  500  16501
    stachyose                  87   8000  500  18772
    AT-AT-CG-CG               118   2000  500   7653
    buckyball-catcher         148    600  100   5402
    double-walled_nanotube    370    800  100   4132

Energies stay in kcal/mol and forces in kcal/mol/A (float64).
Output: ``<out_dir>/<system>_{train,val,test}_default.npz``.

Usage (from the repo root, after scripts/data/download_md22.sh)::

    python scripts/data/prepare_md22.py --raw_dir data/raw/md22 --out_dir data/md22
    python scripts/data/verify_data.py --only data/md22      # array-level check vs. our files
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _prep_common import import_data_module, setup_logging

EXPECTED = {   # (atoms, train, val, test) with the paper's settings
    "Ac-Ala3-NHMe": (42, 6000, 500, 78609),
    "DHA": (56, 8000, 500, 61253),
    "AT-AT": (60, 3000, 500, 16501),
    "stachyose": (87, 8000, 500, 18772),
    "AT-AT-CG-CG": (118, 2000, 500, 7653),
    "buckyball-catcher": (148, 600, 100, 5402),
    "double-walled_nanotube": (370, 800, 100, 4132),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw_dir", default="data/raw/md22",
                    help="directory with md22_<system>.npz (default: data/raw/md22)")
    ap.add_argument("--out_dir", default="data/md22", help="output directory (default: data/md22)")
    ap.add_argument("--molecules", nargs="+", default=list(EXPECTED), choices=list(EXPECTED),
                    metavar="SYSTEM", help="subset of systems (default: all 7)")
    ap.add_argument("--seed", type=int, default=1, help="split seed (paper: 1)")
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = ap.parse_args()
    setup_logging()

    published = args.seed == 1
    if not published:
        print("NOTE: non-default seed; output files are still named *_default.npz, "
              "so use a separate --out_dir.")

    MD22Processor = import_data_module("md22").MD22Processor
    proc = MD22Processor(args.raw_dir, args.out_dir, seed=args.seed)

    out = Path(args.out_dir)
    ok = True
    print(f"{'system':<24}{'atoms':>6}{'train':>7}{'val':>6}{'test':>7}")
    for mol in args.molecules:
        paths = [out / f"{mol}_{s}_default.npz" for s in ("train", "val", "test")]
        if all(p.exists() for p in paths) and not args.force:
            print(f"{mol:<24}  exists, skipped (use --force to rebuild)")
            continue
        info = proc.process_molecule(mol)
        got = (info["n_atoms"], info["n_train"], info["n_val"], info["n_test"])
        flag = ""
        if published and got != EXPECTED[mol]:
            flag, ok = f"   <-- expected {EXPECTED[mol]}", False
        print(f"{mol:<24}{got[0]:>6}{got[1]:>7}{got[2]:>6}{got[3]:>7}{flag}")

    print(f"\nwrote {out}/<system>_{{train,val,test}}_default.npz")
    print("check: python scripts/data/verify_data.py --only data/md22")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
