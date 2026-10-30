"""Shared helpers for the aggregation scripts (standard library only).

The inputs are the JSON files written by ``evaluate.py --out``: lists of result
records with keys tag, molecule, split, seed, label_mode, checkpoint and the
pooled metrics (e_mae_meV, f_mae_meVA, ...), optionally ``per_subset``.
"""

import json
import math
import statistics
import sys


def load_records(paths):
    """Concatenate the records of one or more evaluate.py JSON files."""
    records = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("records", [data])
        for r in data:
            if "tag" not in r or "molecule" not in r:
                raise ValueError(f"{path}: record without tag/molecule (not evaluate.py output?)")
            r.setdefault("_source", path)
            records.append(r)
    return records


def select(records, split="test", label_mode="any", tags=None):
    """Filter records; drop duplicates (same tag/molecule/seed/checkpoint/label
    mode/test file), keeping the last."""
    out = [r for r in records
           if (split == "any" or r.get("split") == split)
           and (label_mode == "any" or r.get("label_mode") == label_mode)
           and (tags is None or r["tag"] in tags)]
    seen = {}
    for r in out:
        key = (r["tag"], r["molecule"], r.get("seed"), r.get("checkpoint"),
               r.get("label_mode"), str(r.get("data", "")).replace("\\", "/").split("/")[-1])
        if key in seen:
            warn(f"duplicate record for {key[:3]}; keeping the last one "
                 f"({r['_source']})")
        seen[key] = r
    out = list(seen.values())
    for tag in dict.fromkeys(r["tag"] for r in out):
        modes = {r.get("label_mode") for r in out if r["tag"] == tag}
        if len(modes) > 1:
            warn(f"tag {tag!r} mixes label modes {sorted(map(str, modes))}; "
                 "use --label_mode to pick one")
    return out


def group(records):
    """{tag: {molecule: [records sorted by seed]}} in first-appearance order."""
    g = {}
    for r in records:
        g.setdefault(r["tag"], {}).setdefault(r["molecule"], []).append(r)
    for mols in g.values():
        for mol, rs in mols.items():
            seeds = [r.get("seed") for r in rs]
            if len(set(seeds)) != len(seeds):
                warn(f"{rs[0]['tag']}/{mol}: several records share a seed {seeds}; "
                     "they are averaged as if independent")
            rs.sort(key=lambda r: (str(r.get("seed")),))
    return g


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def mean_std(values):
    """(mean, sample std or None, n) of the non-None values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    m = statistics.fmean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else None
    return m, s, len(vals)


def fmt(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def fmt_ms(m, s, nd=3):
    if m is None:
        return "–"
    return f"{m:.{nd}f}" if s is None else f"{m:.{nd}f} ± {s:.{nd}f}"


def pct(x, nd=1):
    return "–" if x is None else f"{x:+.{nd}f}%"


def outcome(ours, baseline, decimals):
    """'W' (ours lower), 'T' (equal at the baseline's printed precision) or 'L'."""
    if ours is None or baseline is None:
        return None
    o, b = round(ours, decimals), round(baseline, decimals)
    return "W" if o < b else ("T" if o == b else "L")


def sign_test_p(k, n):
    """Two-sided exact binomial sign test p-value for k 'successes' out of n."""
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def seedwise_average(mols, metric, molecules):
    """Mean over molecules computed per seed; returns (mean, std, n_seeds) when
    every molecule has the same seed set with >= 2 seeds, else (mean of per-
    molecule means, None, n)."""
    seed_sets = [tuple(sorted(str(r.get("seed")) for r in mols[m])) for m in molecules]
    means = [mean_std([r.get(metric) for r in mols[m]])[0] for m in molecules]
    if any(v is None for v in means):
        return None, None, 0
    overall = statistics.fmean(means)
    if len(set(seed_sets)) == 1 and len(seed_sets[0]) > 1:
        per_seed = []
        for seed in seed_sets[0]:
            vals = [next(r.get(metric) for r in mols[m] if str(r.get("seed")) == seed)
                    for m in molecules]
            per_seed.append(statistics.fmean(vals))
        return overall, statistics.stdev(per_seed), len(per_seed)
    return overall, None, min(len(mols[m]) for m in molecules)


def ablation_summary(base, variant, molecules, metric, label_base, label_var, nd=3,
                     metric_label=None):
    """Per-molecule base-vs-variant table + summary statistics (markdown)."""
    rows, gaps, worse, noisy = [], [], 0, 0
    for m in molecules:
        bm, bs, bn = mean_std([r.get(metric) for r in base[m]])
        vm, vs, vn = mean_std([r.get(metric) for r in variant[m]])
        gap = (vm - bm) / bm * 100 if bm else None
        flag = "–"
        if bs is not None and vs is not None:
            se = math.sqrt(bs ** 2 / bn + vs ** 2 / vn)
            clear = abs(vm - bm) > 2 * se
            flag = "clear" if clear else "within noise"
            noisy += not clear
        gaps.append(gap)
        worse += vm > bm
        rows.append([m, fmt_ms(bm, bs, nd), fmt_ms(vm, vs, nd), pct(gap), flag])
    b_avg = statistics.fmean(mean_std([r.get(metric) for r in base[m]])[0] for m in molecules)
    v_avg = statistics.fmean(mean_std([r.get(metric) for r in variant[m]])[0] for m in molecules)
    rows.append(["**mean over molecules**", fmt(b_avg, nd), fmt(v_avg, nd),
                 pct((v_avg - b_avg) / b_avg * 100), ""])
    n = len(molecules)
    g_mean = statistics.fmean(gaps)
    g_sem = statistics.stdev(gaps) / math.sqrt(n) if n > 1 else None
    p = sign_test_p(worse, n)
    text = md_table(["Molecule", label_base, label_var, f"{label_var} vs {label_base}",
                     "gap vs seed noise"], rows)
    text += (f"\n\n- {label_var} worse than {label_base} on {worse}/{n} molecules "
             f"(two-sided sign test p = {p:.4f}).")
    text += (f"\n- Relative reduction of the molecule-averaged {metric_label or metric} "
             f"by {label_base}: "
             f"{(v_avg - b_avg) / v_avg * 100:.1f}% ({fmt(v_avg, nd)} -> {fmt(b_avg, nd)}).")
    text += (f"\n- Per-molecule gap: mean {g_mean:+.1f}%"
             + (f" ± {g_sem:.1f}% (SEM across molecules)" if g_sem is not None else "")
             + f", median {statistics.median(gaps):+.1f}%, "
               f"range {min(gaps):+.1f}% .. {max(gaps):+.1f}%.")
    if noisy:
        text += (f"\n- {noisy} molecule(s) have |gap| <= 2 x combined standard error "
                 "of the seed means ('within noise').")
    return text
