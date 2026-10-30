#!/usr/bin/env python3
"""Build the rMD17 train/val/test NPZs used by every ATOMTRICK rMD17 run.

Thin CLI over :class:`AtomTrick.data.rmd17.RMD17Processor` with the settings of
the reported runs:

* official split ``index_{train,test}_01.csv`` (``--split_id 1``): 1000 train
  indices and 1000 test indices per molecule;
* the 1000 training indices are shuffled with ``np.random.RandomState(0)``
  (``--seed 0``) and the last 5% (50) become the validation set
  (``--val_fraction 0.05``), leaving 950 for training;
* test = the 1000 official test frames, in file order.

Energies stay in kcal/mol and forces in kcal/mol/A (float64); the training
loader converts to eV.  Output: ``<out_dir>/<molecule>_{train,val,test}_default.npz``.

Usage (from the repo root, after scripts/data/download_rmd17.sh)::

    python scripts/data/prepare_rmd17.py --raw_dir data/raw/rmd17 --out_dir data/rmd17
    python scripts/data/verify_data.py --only data/rmd17     # array-level check vs. our files

Re-running with the defaults reproduces our arrays bit for bit (verify_data.py);
the file md5s differ only because numpy stamps zip members with the current time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _prep_common import import_data_module, setup_logging

MOLECULES = ["aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
             "naphthalene", "paracetamol", "salicylic", "toluene", "uracil"]
EXPECTED = (950, 50, 1000)   # train / val / test with the paper's settings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw_dir", default="data/raw/rmd17",
                    help="rMD17 root containing npz_data/ and splits/ (default: data/raw/rmd17)")
    ap.add_argument("--out_dir", default="data/rmd17", help="output directory (default: data/rmd17)")
    ap.add_argument("--molecules", nargs="+", default=MOLECULES, choices=MOLECULES, metavar="MOL",
                    help="subset of molecules (default: all 10)")
    ap.add_argument("--split_id", type=int, default=1, help="official split 1..5 (paper: 1)")
    ap.add_argument("--seed", type=int, default=0, help="val carve-out seed (paper: 0)")
    ap.add_argument("--val_fraction", type=float, default=0.05,
                    help="fraction of the 1000 training indices used for validation (paper: 0.05)")
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = ap.parse_args()
    setup_logging()

    published = (args.split_id, args.seed, args.val_fraction) == (1, 0, 0.05)
    if not published:
        print("NOTE: non-default split settings; output files are still named *_default.npz, "
              "so use a separate --out_dir to avoid mixing them with the paper's splits.")

    RMD17Processor = import_data_module("rmd17").RMD17Processor
    proc = RMD17Processor(args.raw_dir, args.out_dir, seed=args.seed,
                          split_id=args.split_id, val_fraction=args.val_fraction)

    out = Path(args.out_dir)
    ok = True
    print(f"{'molecule':<15}{'atoms':>6}{'train':>7}{'val':>6}{'test':>7}")
    for mol in args.molecules:
        paths = [out / f"{mol}_{s}_default.npz" for s in ("train", "val", "test")]
        if all(p.exists() for p in paths) and not args.force:
            print(f"{mol:<15}  exists, skipped (use --force to rebuild)")
            continue
        info = proc.process_molecule(mol)
        counts = (info["n_train"], info["n_val"], info["n_test"])
        flag = ""
        if published and counts != EXPECTED:
            flag, ok = "   <-- expected 950/50/1000", False
        print(f"{mol:<15}{info['n_atoms']:>6}{counts[0]:>7}{counts[1]:>6}{counts[2]:>7}{flag}")

    print(f"\nwrote {out}/<molecule>_{{train,val,test}}_default.npz")
    print("check: python scripts/data/verify_data.py --only data/rmd17")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
