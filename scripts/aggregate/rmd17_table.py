"""rMD17 tables from evaluate.py JSON results (markdown).

Groups records by (tag, molecule), averages over seeds (mean ± sample std),
compares the main tag with the published baselines (win/tie/loss counts at the
baselines' printed precision, relative gaps), and summarises ablations such as
O(2) vs SO(2) (per-molecule gaps, relative reduction of the molecule-averaged
force MAE, two-sided sign test).

    python scripts/aggregate/rmd17_table.py results/rmd17_*.json \\
        --main atomtrick --compare atomtrick:so2 --out results/rmd17_tables.md

Records are filtered to ``--split test`` by default (use ``--split val`` for
validation-set tables, ``any`` to disable the filter).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B  # noqa: E402
from _common import (ablation_summary, fmt, fmt_ms, group, load_records,  # noqa: E402
                     md_table, mean_std, outcome, pct, seedwise_average, select,
                     sign_test_p, warn)

METRICS = [("e_mae_meV", "E MAE (meV)"), ("f_mae_meVA", "F MAE (meV/Å)"),
           ("f_l2_meVA", "F L2 (meV/Å)")]


def ordered(mols):
    known = [m for m in B.RMD17_MOLECULES if m in mols]
    return known + sorted(m for m in mols if m not in known)


def per_tag_table(tag, mols):
    molecules = ordered(mols)
    rows = []
    for m in molecules:
        cells = [m, str(len(mols[m]))]
        for key, _ in METRICS:
            mu, sd, _n = mean_std([r.get(key) for r in mols[m]])
            cells.append(fmt_ms(mu, sd))
        rows.append(cells)
    cells = ["**mean over molecules**", ""]
    for key, _ in METRICS:
        mu, sd, _n = seedwise_average(mols, key, molecules)
        cells.append(fmt_ms(mu, sd))
    rows.append(cells)
    seeds = sorted({str(r.get("seed")) for rs in mols.values() for r in rs})
    head = (f"### {tag}: per-molecule MAE (mean ± sample std over seeds {', '.join(seeds)}; "
            "± on the mean row = std across seeds of the molecule-averaged value)\n\n")
    return head + md_table(["Molecule", "n"] + [lbl for _, lbl in METRICS], rows)


def baseline_table(tag, mols, quantity, metric, models, gap_vs):
    molecules = [m for m in B.RMD17_MOLECULES if m in mols]
    ours = {m: mean_std([r.get(metric) for r in mols[m]])[0] for m in molecules}
    wtl = {n: {"W": 0, "T": 0, "L": 0} for n in models}
    rows = []
    for m in molecules:
        vals = {n: B.RMD17[n][quantity][m] for n in models}
        best = min([v for v in vals.values()] + [ours[m]])
        cells = [m]
        for n in models:
            d = B.value_decimals(vals[n], B.RMD17[n], m)
            txt = fmt(vals[n], d)
            cells.append(f"**{txt}**" if vals[n] == best else txt)
            res = outcome(ours[m], vals[n], d)
            if res:
                wtl[n][res] += 1
        txt = fmt(ours[m], 3)
        cells.append(f"**{txt}**" if ours[m] == best else txt)
        g = B.RMD17[gap_vs][quantity][m] if gap_vs else None
        cells.append(pct((ours[m] - g) / g * 100) if g else "–")
        rows.append(cells)
    mean_row = ["mean"] + [fmt(sum(B.RMD17[n][quantity][m] for m in molecules) / len(molecules), 3)
                           for n in models]
    mean_row += [fmt(sum(ours.values()) / len(molecules), 3), ""]
    rows.append(mean_row)
    rows.append([f"{tag} W/T/L vs"] + [f"{wtl[n]['W']}/{wtl[n]['T']}/{wtl[n]['L']}"
                                        for n in models] + ["", ""])
    unit = "meV/Å" if quantity == "force" else "meV"
    head = (f"### {quantity.capitalize()} MAE ({unit}): {tag} (seed mean) vs published "
            f"baselines\n\nW/T/L = {tag} lower / equal / higher at each baseline's printed "
            f"precision. Bold = lowest in the row.\n\n")
    table = md_table(["Molecule"] + models + [tag, f"Δ vs {gap_vs}" if gap_vs else "Δ"], rows)
    if len(molecules) < len(B.RMD17_MOLECULES):
        table += f"\n\n(only {len(molecules)}/10 molecules available for {tag})"
    return head + table


def pick_main(tags, requested=None):
    if requested:
        return requested
    for t in tags:
        if t == "rmd17_atomtrick":
            return t
    for t in tags:
        if "atomtrick" in t:
            return t
    return tags[0]


def default_pairs(tags, main_tag):
    """Every other tag against the main tag."""
    return [(main_tag, t) for t in tags if t != main_tag]


def overview_table(g, pairs):
    """Table-3-style summary: molecule-averaged metrics per variant vs its base."""
    rows = []
    for base, var in pairs:
        common = [m for m in ordered(g[base]) if m in g[var]]
        if not common:
            continue
        cells = [var, base, str(len(common)),
                 str(min(len(g[var][m]) for m in common))]
        for key, _ in METRICS:
            mu, sd, _n = seedwise_average(g[var], key, common)
            cells.append(fmt_ms(mu, sd))
        bf = seedwise_average(g[base], "f_mae_meVA", common)[0]
        vf = seedwise_average(g[var], "f_mae_meVA", common)[0]
        worse = sum(mean_std([r.get("f_mae_meVA") for r in g[var][m]])[0]
                    > mean_std([r.get("f_mae_meVA") for r in g[base][m]])[0] for m in common)
        p = sign_test_p(worse, len(common))
        cells += [pct((vf - bf) / bf * 100) if bf and vf else "–",
                  f"{worse}/{len(common)} (p = {p:.3f})"]
        rows.append(cells)
    head = ("### Ablation overview (molecule-averaged; ± = std across seeds of the "
            "molecule average)\n\n")
    return head + md_table(["Variant", "Base", "molecules", "seeds"]
                           + [lbl for _, lbl in METRICS]
                           + ["ΔF vs base", "F worse than base (sign test)"], rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="evaluate.py JSON output file(s)")
    ap.add_argument("--split", default="test", help="test (default) | val | any")
    ap.add_argument("--label_mode", default="any", help="float64 | any (records of this release are float64)")
    ap.add_argument("--tags", nargs="+", default=None, help="tags to include (default all)")
    ap.add_argument("--main", default=None,
                    help="tag compared with the baselines (default: 'rmd17_atomtrick', else "
                         "the first tag containing 'atomtrick', else the first tag)")
    ap.add_argument("--baselines", nargs="+", default=list(B.RMD17),
                    choices=list(B.RMD17), help="baseline columns")
    ap.add_argument("--gap_vs", default="GotenNet",
                    help="baseline for the relative-gap column ('' to disable)")
    ap.add_argument("--compare", nargs="+", default=None, metavar="BASE:VARIANT",
                    help="ablation pairs for the overview, e.g. rmd17_atomtrick:rmd17_so2 "
                         "(default: every other tag vs main)")
    ap.add_argument("--detail", nargs="+", default=None, metavar="VARIANT",
                    help="variants that get per-molecule ablation tables (default: tags "
                         "containing 'so2')")
    ap.add_argument("--out", default=None, help="also write the markdown here")
    args = ap.parse_args(argv)

    recs = select(load_records(args.results), args.split, args.label_mode, args.tags)
    if not recs:
        raise SystemExit(f"no records with split={args.split!r} label_mode={args.label_mode!r}")
    g = group(recs)
    main_tag = pick_main(list(g), args.main)
    if main_tag not in g:
        raise SystemExit(f"--main {main_tag!r} not among tags {list(g)}")

    parts = [f"## rMD17 ({args.split} split)"]
    for tag, mols in g.items():
        parts.append(per_tag_table(tag, mols))
    gap_vs = args.gap_vs or None
    parts.append(baseline_table(main_tag, g[main_tag], "force", "f_mae_meVA",
                                args.baselines, gap_vs))
    parts.append(baseline_table(main_tag, g[main_tag], "energy", "e_mae_meV",
                                args.baselines, gap_vs))

    if args.compare is not None:
        pairs = []
        for pair in args.compare:
            base, _, var = pair.partition(":")
            if base not in g or var not in g:
                warn(f"--compare {pair}: tag not found (have {list(g)})")
                continue
            pairs.append((base, var))
    else:
        pairs = default_pairs(list(g), main_tag)
    if pairs:
        parts.append(overview_table(g, pairs))
    detail = args.detail
    if detail is None:
        detail = [v for _, v in pairs if "so2" in v.lower()]
    for base, var in pairs:
        if var not in detail:
            continue
        common = [m for m in ordered(g[base]) if m in g[var]]
        if len(common) < len(g[base]):
            warn(f"{base}:{var}: comparing on the {len(common)} molecules both tags have")
        for metric, label in (("f_mae_meVA", "force MAE (meV/Å)"),
                              ("e_mae_meV", "energy MAE (meV)")):
            parts.append(f"### Ablation {base} vs {var}: {label}\n\n"
                         + ablation_summary(g[base], g[var], common, metric, base, var,
                                            metric_label=label))

    text = "\n\n".join(parts) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
