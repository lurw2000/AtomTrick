"""MD22 tables from evaluate.py JSON results (markdown).

Per-system test errors of each tag (mean ± sample std over seeds when several
exist), then the main tag against the published baselines (the paper's
MD22 columns by default), with win/tie/loss counts at each baseline's
printed precision and the relative gap to GotenNet-B.

    python scripts/aggregate/md22_table.py results/md22_test.json --out results/md22_tables.md

The baseline comparison uses the main tag for every system; ``--use MOLECULE=TAG``
takes a system from another tag instead (marked with a dagger).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B  # noqa: E402
from _common import (fmt, fmt_ms, group, load_records, md_table, mean_std,  # noqa: E402
                     outcome, pct, select)

METRICS = [("e_mae_meV", "E MAE (meV)", 2), ("e_mae_meV_atom", "E MAE (meV/atom)", 4),
           ("f_mae_meVA", "F MAE (meV/Å)", 3), ("f_rmse_meVA", "F RMSE (meV/Å)", 3),
           ("f_l2_meVA", "F L2 (meV/Å)", 3)]


def ordered(mols):
    known = [m for m in B.MD22_MOLECULES if m in mols]
    return known + sorted(m for m in mols if m not in known)


def per_tag_table(tag, mols):
    rows = []
    for m in ordered(mols):
        rs = mols[m]
        n_struct = rs[0].get("n_structures")
        cells = [m, str(B.MD22_ATOMS.get(m, "")), str(n_struct), str(len(rs))]
        for key, _, nd in METRICS:
            mu, sd, _n = mean_std([r.get(key) for r in rs])
            cells.append(fmt_ms(mu, sd, nd))
        rows.append(cells)
    return (f"### {tag}: MD22 test errors (mean ± sample std over seeds where n > 1)\n\n"
            + md_table(["System", "atoms", "test N", "n"] + [l for _, l, _ in METRICS], rows))


def pick_main(tags, requested=None):
    if requested:
        return requested
    for t in tags:
        if t == "md22_atomtrick":
            return t
    return next((t for t in tags if "atomtrick" in t), tags[0])


def substitutions(g, main_tag, use):
    """{molecule: tag} for systems taken from another tag in the comparison."""
    subs = {}
    if use is None:
        return subs
    for item in use:
        if item == "none":
            return {}
        mol, _, tag = item.partition("=")
        if tag not in g or mol not in g[tag]:
            raise SystemExit(f"--use {item}: no results for {mol} under tag {tag!r}")
        subs[mol] = tag
    return subs


def baseline_table(tag, mols, quantity, metric, models, gap_vs, subs=None):
    subs = subs or {}
    molecules = [m for m in B.MD22_MOLECULES if m in mols]
    ours = {m: mean_std([r.get(metric) for r in mols[m]])[0] for m in molecules}
    wtl = {n: {"W": 0, "T": 0, "L": 0} for n in models}
    rows = []
    for m in molecules:
        vals = {n: B.MD22[n][quantity][m] for n in models}
        present = [v for v in vals.values() if v is not None] + [ours[m]]
        best = min(present)
        cells = [m]
        for n in models:
            v = vals[n]
            d = B.value_decimals(v, B.MD22[n], m) if v is not None else 2
            txt = fmt(v, d)
            cells.append(f"**{txt}**" if v is not None and v == best else txt)
            res = outcome(ours[m], v, d)
            if res:
                wtl[n][res] += 1
        txt = fmt(ours[m], 2) + ("†" if m in subs else "")
        cells.append(f"**{txt}**" if ours[m] == best else txt)
        g = B.MD22[gap_vs][quantity][m] if gap_vs else None
        cells.append(pct((ours[m] - g) / g * 100) if g else "–")
        rows.append(cells)
    rows.append([f"{tag} W/T/L vs"] + [f"{wtl[n]['W']}/{wtl[n]['T']}/{wtl[n]['L']}"
                                        for n in models] + ["", ""])
    unit = "meV/Å" if quantity == "force" else "meV"
    head = (f"### MD22 {quantity} MAE ({unit}): {tag} vs published baselines\n\n"
            "W/T/L = lower / equal / higher at the baseline's printed precision "
            "(– = not reported). Bold = lowest in the row.\n\n")
    table = md_table(["System"] + models + [tag, f"Δ vs {gap_vs}" if gap_vs else "Δ"], rows)
    for m, t in subs.items():
        table += f"\n\n† {m}: result of tag `{t}`."
    return head + table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="evaluate.py JSON output file(s)")
    ap.add_argument("--split", default="test", help="test (default) | val | any")
    ap.add_argument("--label_mode", default="any", help="float64 | any (records of this release are float64)")
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--main", default=None, help="tag compared with the baselines "
                                                 "(default: md22_atomtrick / first "
                                                 "'atomtrick' tag / first tag)")
    ap.add_argument("--use", nargs="+", default=None, metavar="MOLECULE=TAG",
                    help="take these systems from another tag in the baseline comparison "
                         "('none' = no substitution, the default)")
    ap.add_argument("--baselines", nargs="+", default=B.MD22_PAPER_COLUMNS,
                    choices=list(B.MD22))
    ap.add_argument("--gap_vs", default="GotenNet-B",
                    help="baseline for the relative-gap column ('' to disable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    recs = select(load_records(args.results), args.split, args.label_mode, args.tags)
    if not recs:
        raise SystemExit(f"no records with split={args.split!r} label_mode={args.label_mode!r}")
    g = group(recs)
    main_tag = pick_main(list(g), args.main)
    if main_tag not in g:
        raise SystemExit(f"--main {main_tag!r} not among tags {list(g)}")
    subs = substitutions(g, main_tag, args.use)
    compared = dict(g[main_tag])
    compared.update({m: g[t][m] for m, t in subs.items()})
    parts = [f"## MD22 ({args.split} split)"]
    parts += [per_tag_table(t, mols) for t, mols in g.items()]
    gap_vs = args.gap_vs or None
    parts.append(baseline_table(main_tag, compared, "force", "f_mae_meVA",
                                args.baselines, gap_vs, subs))
    parts.append(baseline_table(main_tag, compared, "energy", "e_mae_meV",
                                args.baselines, gap_vs, subs))
    text = "\n\n".join(parts) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
