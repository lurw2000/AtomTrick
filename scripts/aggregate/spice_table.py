"""SPICE (MACE-OFF23 test) table from evaluate.py JSON results (markdown).

Lays out our per-subset errors next to MACE-OFF23-S/M/L (SI Table S1): energy
MAE/RMSE in meV/atom, force MAE/RMSE in meV/Å, one row per (subset, metric),
plus the structure-pooled OVERALL row (MACE-OFF23 reports none), and counts
the cells where each tag beats each MACE-OFF23 model (strictly lower at the
2-decimal precision of Table S1; ties counted separately).

Records must come from ``evaluate.py --per_subset``.

    python scripts/aggregate/spice_table.py results/spice_test.json --out results/spice_table.md
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B  # noqa: E402
from _common import (fmt, fmt_ms, load_records, md_table, mean_std, outcome,  # noqa: E402
                     select)

LABELS = {"e_mae_meV_atom": "E MAE (meV/atom)", "e_rmse_meV_atom": "E RMSE (meV/atom)",
          "f_mae_meVA": "F MAE (meV/Å)", "f_rmse_meVA": "F RMSE (meV/Å)"}
DIGITS = {"e_mae_meV_atom": 3, "e_rmse_meV_atom": 3, "f_mae_meVA": 2, "f_rmse_meVA": 2}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="evaluate.py JSON output file(s)")
    ap.add_argument("--split", default="test", help="test (default) | val | any")
    ap.add_argument("--label_mode", default="any", help="float64 | any (records of this release are float64)")
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--baselines", nargs="+", default=list(B.SPICE), choices=list(B.SPICE))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    recs = [r for r in select(load_records(args.results), args.split, args.label_mode,
                              args.tags) if r.get("per_subset")]
    if not recs:
        raise SystemExit("no records with per_subset metrics (run evaluate.py --per_subset)")
    # One column per (tag, test file): results of one run on different test files
    # stay separate; seeds of one tag on one file are averaged.
    def stem(r):
        b = os.path.basename(str(r.get("data")))
        return b[:-4] if b.endswith(".npz") else b
    keys = list(dict.fromkeys((r["tag"], stem(r)) for r in recs))
    multi = {t for t in {k[0] for k in keys} if sum(k[0] == t for k in keys) > 1}
    tags = [f"{t} [{f}]" if t in multi else t for t, f in keys]
    by_tag = {label: [r for r in recs if (r["tag"], stem(r)) == key]
              for label, key in zip(tags, keys)}

    subsets = list(B.SPICE_SUBSETS)
    extra = sorted({s for r in recs for s in r["per_subset"]} - set(subsets))
    subsets += extra

    def ours(tag, subset, key):
        vals = [(r if subset == "OVERALL" else r["per_subset"].get(subset, {})).get(key)
                for r in by_tag[tag]]
        return mean_std(vals)

    wins = {(t, b): {"W": 0, "T": 0, "L": 0} for t in tags for b in args.baselines}
    rows = []
    for s in subsets + ["OVERALL"]:
        for key, label in LABELS.items():
            cells = [f"**{s}**" if s == "OVERALL" else s, label]
            for t in tags:
                mu, sd, _n = ours(t, s, key)
                cells.append(fmt_ms(mu, sd, DIGITS[key]))
            for b in args.baselines:
                v = B.SPICE[b]["subsets"].get(s, {}).get(key)
                cells.append(fmt(v, 2))
                for t in tags:
                    res = outcome(ours(t, s, key)[0], v, 2)
                    if res:
                        wins[(t, b)][res] += 1
            rows.append(cells)
    count_lines = []
    for label, rs in by_tag.items():
        r = rs[0]
        name = label if "[" in label else f"{label} [{stem(r)}]"
        count_lines.append(f"- {name}: " + ", ".join(
            f"{s} {r['per_subset'].get(s, {}).get('n_structures', 0):,}" for s in subsets)
            + f"; total {r.get('n_structures', 0):,}")

    head = ("## SPICE / MACE-OFF23 test set\n\n"
            "Energies per atom, forces per Cartesian component. MACE-OFF23 values: SI "
            "Table S1 (Kovács et al. 2023). OVERALL = pooled over all structures (not "
            "reported by MACE-OFF23).\n\nTest structures per subset:\n\n"
            + "\n".join(count_lines) + "\n")
    table = md_table(["Subset", "Metric"] + tags + args.baselines, rows)
    summary = ["", f"Cells won against each MACE-OFF23 model (out of "
                   f"{len(B.SPICE_SUBSETS) * len(LABELS)} subset x metric cells; W/T/L):", ""]
    summary.append(md_table(["tag"] + args.baselines,
                            [[t] + [f"{wins[(t, b)]['W']}/{wins[(t, b)]['T']}/{wins[(t, b)]['L']}"
                                    for b in args.baselines] for t in tags]))
    text = head + "\n" + table + "\n" + "\n".join(summary) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
