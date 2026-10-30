#!/usr/bin/env python3
"""Merge the per-model latency CSVs (4 baselines, eager and compiled, + ATOMTRICK) into one
long table and a markdown summary (median latency, peak memory and throughput; rMD17 + MD22;
one table per batch size).

Inputs
------
* Baselines, eager (one CSV per model and dataset, written by
  ``baselines/*/benchmark_inference*.py``):
  ``<baseline_dir>/<prefix>_<dataset>_latency_<gpu_tag>.csv`` with prefix in
  {equiformer, equiformer_v2, gotennet, mace}.  Individual files can also be given with
  ``--baseline MODEL:DATASET:PATH``.
* Baselines, compiled (the same harnesses with ``--compile inductor`` / ``--compile fused``),
  read when present: ``<baseline_dir>/<prefix>_compiled_<variant>_<dataset>_latency_<gpu_tag>.csv``.
  Both variants land in the long CSV as ``<label> (compiled: inductor)`` /
  ``<label> (compiled: fused)``; the markdown tables show ONE compiled column per baseline,
  ``<label> (compiled)`` = the variant selected by ``--compiled_variant`` (default ``auto`` =
  the lower geometric-mean median latency over the (dataset, molecule, bs) rows where both
  variants are ok; a single variant with ok rows is taken as is; none -> no compiled column).
  The long CSV marks the selection in its ``selected`` column (1 for every eager and ATOMTRICK
  row and for the selected variant, 0 otherwise; plot.py skips 0) and carries a ``compile``
  column (none / inductor / fused; ATOMTRICK's compiled rows say makefx).
* ATOMTRICK (``bench_atomtrick.py``; has ``dataset`` and ``path`` columns, path in
  {eager, makefx}) -> emitted as two models, ``<label> (eager)`` and ``<label> (compiled)``.
  "compiled" = make_fx + torch.compile (Inductor); there are no hand-written kernels.

All numbers: energy + forces, warmup 10 + 50 timed calls, per-call CUDA-event timing.
Latency = MEDIAN of the per-call times.  Two throughputs are written to the long CSV:
``throughput_samples_per_s`` = bs*1000/mean (spike-sensitive) and
``throughput_median_samples_per_s`` = bs*1000/median (the paper's definition).

Reproduce the committed summary of the reported set with the flags run_all.sh passes
(benchmarks/latency/README.md, "Re-running it"); inputs and outputs have no defaults.
"""

import argparse
import csv
import math
import os

# (model label, csv prefix)
BASELINES = [
    ("Equiformer",   "equiformer"),
    ("EquiformerV2", "equiformer_v2"),
    ("GotenNet",     "gotennet"),
    ("MACE",         "mace"),
]
BASELINE_NAMES = [b[0] for b in BASELINES]
COMPILE_VARIANTS = ("inductor", "fused")
# compile diagnostics written by latency_common.CompileStats (kept per row, not in the long CSV)
DIAG = ["compile_graphs_warmup", "compile_graphs_timed", "compile_graph_breaks",
        "compile_warmup_s", "compile_retimed", "pytorch_jit", "create_graph"]

RMD17_ORDER = ["ethanol", "malonaldehyde", "benzene", "uracil", "toluene",
               "salicylic", "naphthalene", "paracetamol", "aspirin", "azobenzene"]
MD22_ORDER = ["Ac_Ala3_NHMe", "DHA", "AT_AT", "stachyose", "AT_AT_CG_CG",
              "buckyball_catcher", "double_walled_nanotube"]
MOL_ORDER = {("rmd17", m): i for i, m in enumerate(RMD17_ORDER)}
MOL_ORDER.update({("md22", m): 100 + i for i, m in enumerate(MD22_ORDER)})

# Fairness notes printed under the tables (see baselines/README.md for details).  The input
# and force settings are read from the CSV columns of each model's eager rows (`coords`,
# `forces`) when present, so the note describes the measurement that was merged.
NOTES = {
    "Equiformer": "eager, strict fp32; {coords}; radius graph built inside the timed forward",
    "EquiformerV2": "eager, strict fp32; OC20 N@12_L@6_M@2 config, never trained on these "
                    "datasets; {forces}; {coords}; graph built inside the timed forward",
    "GotenNet": "eager, strict fp32; real frames; per-dataset production config; radius graph "
                "built inside the timed forward",
    "MACE": "eager (not compiled), strict fp32, cuEquivariance off; real frames; neighbour "
            "list built outside the timed region",
}
COORDS_TEXT = {
    "raw": "real frames 0..bs-1 of the raw trajectory",
    "real": "real frames of the ATOMTRICK processed test split",
    "synthetic": "SYNTHETIC coordinates (pos ~ N(0,3 A), z ~ U{1..9})",
}
FORCES_TEXT = {
    "autograd": "autograd forces -dE/dR without the direct force head",
    "direct": "DIRECT (non-conservative) force head",
}
# What differs for the compiled column of each baseline, beyond the common compiled protocol.
COMPILED_NOTES = {
    "Equiformer": "e3nn TorchScript codegen off (`jit_script_fx=False`); radius graph still "
                  "built inside the timed forward",
    "EquiformerV2": "the harness's model + autograd.grad closure is the compiled callable; "
                    "graph still built inside the timed forward",
    "GotenNet": "radius graph still built inside the timed forward; one random-init model "
                "instance reused for all molecules of a dataset (the eager column builds one per "
                "molecule; latency is weight-agnostic) so the dynamic-shape graph is shared",
    "MACE": "built as MACE's own `prepare()` builds it (e3nn codegen off, registered modules "
            "fx-simplified; the upstream torch.compile recipe); one random-init model instance "
            "reused for all molecules (the eager column builds one per molecule) so the "
            "dynamic-shape graph is shared; run in a separate Python <= 3.12 copy of the MACE "
            "environment, because torch 2.5.1 has no torch.compile on Python 3.13 (see the "
            "Environments table of MEASUREMENT_CONDITIONS.md)",
}
COMPILED_PROTOCOL = (
    "the same harness, inputs and precision as the eager column, with the whole energy + "
    "force callable wrapped in `torch.compile(dynamic=True)` (TorchInductor, default mode, no "
    "CUDA graphs), parameters frozen (`requires_grad_(False)`, inference deployment), "
    "`PYTORCH_JIT=0`; Dynamo trace + Inductor compile happen in the 10 warm-up calls and are "
    "excluded from the latency. Two recipes were measured per model: `inductor` = Dynamo "
    "graph-breaks at `torch.autograd.grad`, the energy forward is compiled and AOTAutograd's "
    "compiled backward computes the forces (create_graph=False, as eager); `fused` = "
    "`torch.autograd.grad` allowed in the graph, so forward + force backward form ONE compiled "
    "graph (MACE's upstream recipe, generalised; create_graph=True inside, traced with grad "
    "enabled, outputs detached inside the compiled callable so AOTAutograd compiles an "
    "inference graph that runs under no_grad -- the shape of ATOMTRICK's make_fx step). After "
    "the warm-up each compiled row's energy and forces were checked against one eager call of "
    "the same model (max relative error recorded; a mismatch is a compile_error)")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fint(x):
    v = fnum(x)
    return None if v is None else int(v)


def row_ok(r, key):
    return r is not None and r["status"] == "ok" and r[key] is not None


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atomtrick_csv", required=True,
                    help="bench_atomtrick.py output, e.g. results/h100_fp32_2026-10-20/atomtrick_latency.csv")
    ap.add_argument("--baseline_dir", required=True,
                    help="directory of the baseline harness CSVs, e.g. results/h100_fp32_2026-10-20/raw")
    ap.add_argument("--gpu_tag", default="h100",
                    help="suffix of the baseline CSV names (<prefix>_<dataset>_latency_<tag>.csv)")
    ap.add_argument("--baseline", action="append", default=[],
                    help="extra/override baseline CSV as MODEL:DATASET:PATH (repeatable)")
    ap.add_argument("--compiled_variant", choices=["auto", "inductor", "fused"], default="auto",
                    help="which compiled-baseline variant the markdown tables show as "
                         "'<model> (compiled)': auto = per model the lower geometric-mean median "
                         "latency over the rows where both variants are ok (default)")
    ap.add_argument("--atomtrick_label", default="ATOMTRICK")
    ap.add_argument("--compiled_suffix", default="compiled",
                    help="label suffix for the make_fx + torch.compile rows")
    ap.add_argument("--atomtrick_note", default=None,
                    help="free-text precision note for the ATOMTRICK rows (default: derived "
                         "from the tf32 column if present, else 'TF32 matmuls')")
    ap.add_argument("--out_long", required=True, help="output: the merged long CSV")
    ap.add_argument("--out_md", required=True, help="output: the markdown summary")
    ap.add_argument("--title_gpu", default="H100 80GB")
    args = ap.parse_args()

    at_eager = f"{args.atomtrick_label} (eager)"
    at_comp = f"{args.atomtrick_label} ({args.compiled_suffix})"

    def comp_label(model, variant):          # long-CSV label of a compiled-baseline variant
        return f"{model} (compiled: {variant})"

    def md_comp_label(model):                # markdown label of the selected variant
        return f"{model} (compiled)"

    baseline_labels = list(BASELINE_NAMES)   # eager baseline labels, column order
    model_order = []                         # long-CSV sort order
    for m in BASELINE_NAMES:
        model_order += [m] + [comp_label(m, v) for v in COMPILE_VARIANTS]
    model_order += [at_eager, at_comp]
    rows = []
    recorded = {}                            # eager-row settings per model: {col: set(values)}

    def add(dataset, model, molecule, atoms, bs, n_params, median, mean, mem, gpu, status,
            compile_="none", diag=None):
        bs = int(bs)
        rows.append({
            "dataset": dataset, "model": model, "molecule": molecule,
            "atoms": int(float(atoms)) if atoms not in (None, "") else None,
            "batch_size": bs,
            "n_params": int(float(n_params)) if n_params not in (None, "") else None,
            "latency_median_ms": median, "latency_mean_ms": mean,
            "throughput_samples_per_s": bs * 1000.0 / mean if mean else None,
            "throughput_median_samples_per_s": bs * 1000.0 / median if median else None,
            "peak_mem_gb": mem, "gpu": gpu, "status": status,
            "compile": compile_, "selected": 1,
            "_diag": diag or {}, "_md": model,
        })

    def read_baseline(model, dataset, path, variant=None):
        if not os.path.exists(path):
            if variant is None:
                print(f"WARN missing: {path}")
            return
        with open(path) as f:
            for r in csv.DictReader(f):
                if variant is None:
                    for col in ("coords", "forces"):
                        if r.get(col):
                            recorded.setdefault(model, {}).setdefault(col, set()).add(r[col])
                add(dataset, model if variant is None else comp_label(model, variant),
                    r["molecule"], r.get("atoms"), r["batch_size"],
                    r.get("n_params"), fnum(r.get("latency_median_ms")),
                    fnum(r.get("latency_mean_ms")), fnum(r.get("peak_mem_gb")),
                    r.get("gpu_name"), r.get("status"),
                    compile_=variant or "none",
                    diag=None if variant is None else {k: r.get(k) for k in DIAG})

    # ---- baselines ----
    overrides = {}
    for spec in args.baseline:
        model, dataset, path = spec.split(":", 2)
        overrides[(model, dataset)] = path
    for model, prefix in BASELINES:
        for dataset in ("rmd17", "md22"):
            path = overrides.pop((model, dataset), os.path.join(
                args.baseline_dir, f"{prefix}_{dataset}_latency_{args.gpu_tag}.csv"))
            read_baseline(model, dataset, path)
            for variant in COMPILE_VARIANTS:
                read_baseline(model, dataset, os.path.join(
                    args.baseline_dir,
                    f"{prefix}_compiled_{variant}_{dataset}_latency_{args.gpu_tag}.csv"), variant)
    for (model, dataset), path in overrides.items():     # models not in BASELINES
        read_baseline(model, dataset, path)
        if model not in model_order:
            model_order.insert(model_order.index(at_eager), model)
            baseline_labels.append(model)

    # ---- ATOMTRICK (split by path) ----
    tf32_seen = set()
    if os.path.exists(args.atomtrick_csv):
        label = {"eager": at_eager, "makefx": at_comp}
        with open(args.atomtrick_csv) as f:
            for r in csv.DictReader(f):
                model = label.get(r["path"], f"{args.atomtrick_label} ({r['path']})")
                if r.get("tf32"):
                    tf32_seen.add(r["tf32"])
                add(r["dataset"], model, r["molecule"], r.get("atoms"), r["batch_size"],
                    r.get("n_params"), fnum(r.get("latency_median_ms")),
                    fnum(r.get("latency_mean_ms")), fnum(r.get("peak_mem_gb")),
                    r.get("gpu_name"), r.get("status"),
                    compile_="none" if r["path"] == "eager" else r["path"])
    else:
        print(f"WARN missing ATOMTRICK CSV: {args.atomtrick_csv}")

    # ---- compiled-baseline variant selection ----
    def rkey(r):
        return (r["dataset"], r["molecule"], r["batch_size"])

    comp_rows = {m: {v: [r for r in rows if r["model"] == comp_label(m, v)]
                     for v in COMPILE_VARIANTS} for m in baseline_labels}
    selected, variant_stats = {}, {}
    for m in baseline_labels:
        okl = {v: {rkey(r): r["latency_median_ms"] for r in comp_rows[m][v]
                   if row_ok(r, "latency_median_ms")} for v in COMPILE_VARIANTS}
        common = sorted(set(okl["inductor"]) & set(okl["fused"]))
        ratio = geomean([okl["fused"][k] / okl["inductor"][k] for k in common])
        variant_stats[m] = {"common": len(common), "fused_over_inductor": ratio,
                            "n_ok": {v: len(okl[v]) for v in COMPILE_VARIANTS}}
        if not any(comp_rows[m].values()):
            continue
        if args.compiled_variant != "auto":
            if okl[args.compiled_variant]:       # a forced variant needs at least one ok row
                selected[m] = args.compiled_variant
            continue
        have = [v for v in COMPILE_VARIANTS if okl[v]]
        if len(have) == 1:
            selected[m] = have[0]
        elif len(have) == 2:
            if common:
                gm = {v: geomean([okl[v][k] for k in common]) for v in COMPILE_VARIANTS}
                selected[m] = min(COMPILE_VARIANTS, key=lambda v: (gm[v], v != "inductor"))
            else:                                   # no common ok row: the one with more ok rows
                selected[m] = max(COMPILE_VARIANTS,
                                  key=lambda v: (len(okl[v]), v == "inductor"))
    for m in baseline_labels:
        for v in COMPILE_VARIANTS:
            for r in comp_rows[m][v]:
                if selected.get(m) == v:
                    r["_md"] = md_comp_label(m)
                else:
                    r["selected"], r["_md"] = 0, None
    if selected:
        print("compiled baseline variant shown in the tables: "
              + ", ".join(f"{m}: {v}" for m, v in selected.items()))

    def sort_key(r):
        return (MOL_ORDER.get((r["dataset"], r["molecule"]), 999),
                model_order.index(r["model"]) if r["model"] in model_order else 99,
                r["batch_size"])

    rows.sort(key=sort_key)

    cols = ["dataset", "model", "molecule", "atoms", "batch_size", "n_params",
            "latency_median_ms", "latency_mean_ms", "throughput_samples_per_s",
            "throughput_median_samples_per_s", "peak_mem_gb", "gpu", "status",
            "compile", "selected"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_long)), exist_ok=True)
    with open(args.out_long, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out_long}  ({len(rows)} rows)")

    # ---- markdown summary ----
    # column order: per baseline eager then (selected) compiled, then ATOMTRICK eager, compiled
    md_order = []
    for m in baseline_labels:
        md_order.append(m)
        if m in selected:
            md_order.append(md_comp_label(m))
    md_order += [at_eager, at_comp]
    has_comp = bool(selected)                                      # a compiled column exists
    has_comp_rows = any(any(comp_rows[m].values()) for m in baseline_labels)   # files were read
    index = {(r["dataset"], r["_md"], r["molecule"], r["batch_size"]): r for r in rows if r["_md"]}

    def get(dataset, model, molecule, bs):
        return index.get((dataset, model, molecule, bs))

    ok = row_ok

    def cell(dataset, model, molecule, bs, key, fmt):
        r = get(dataset, model, molecule, bs)
        if r is None:
            return "—"
        if ok(r, key):
            return fmt(r)
        return "OOM" if r["status"] == "oom" else "err"

    def atoms_of(dataset, molecule):
        for r in rows:
            if r["dataset"] == dataset and r["molecule"] == molecule and r["atoms"]:
                return r["atoms"]
        return "?"

    def fastest(dataset, mol, bs, labels):
        base = [(get(dataset, m, mol, bs)["latency_median_ms"], m) for m in labels
                if ok(get(dataset, m, mol, bs), "latency_median_ms")]
        comp = get(dataset, at_comp, mol, bs)
        if base and ok(comp, "latency_median_ms"):
            best, bname = min(base)
            return f"{bname} | {best / comp['latency_median_ms']:.2f}x"
        return "— | —"

    datasets = [("rmd17", RMD17_ORDER), ("md22", MD22_ORDER)]
    baselines_present = list(baseline_labels)
    compiled_present = [m for m in baseline_labels if m in selected]
    L = []
    L.append(f"# Inference latency ({args.title_gpu}) — energy + forces")
    L.append("")
    if has_comp:
        which = ("the table shows the faster one" if args.compiled_variant == "auto" else
                 f"the table shows the `{args.compiled_variant}` recipe (forced by "
                 f"`--compiled_variant {args.compiled_variant}`)")
        tail = ("Baselines run in two modes: **eager** = plain PyTorch (the reported protocol), "
                "**`<model> (compiled)`** = the same harness with the whole energy + force "
                "callable under `torch.compile(dynamic=True)` / TorchInductor (parameters frozen, "
                f"`PYTORCH_JIT=0`); of the two compile recipes measured per baseline {which}, "
                "see \"Compiled-baseline variants\".")
    elif has_comp_rows:
        tail = ("Compiled baselines (`torch.compile`) were measured but produced no ok row"
                + ("" if args.compiled_variant == "auto" else
                   f" for the forced `--compiled_variant {args.compiled_variant}`")
                + ", see \"Compiled-baseline variants\"; every baseline column here is eager.")
    else:
        tail = "All baselines run eager."
    L.append("Warmup 10 + 50 timed calls, per-call CUDA-event timing, random-init weights. "
             f"**{at_comp}** = make_fx whole-graph capture of forward + force backward, "
             "compiled by torch.compile / TorchInductor (no hand-written kernels, no CUDA "
             f"graphs); **{at_eager}** = plain PyTorch autograd. " + tail)
    L.append("")

    for bs in (1, 4):
        L.append(f"## Median latency (ms), batch size {bs}")
        L.append("")
        if has_comp:
            L.append("| Dataset | Molecule | Atoms | " + " | ".join(md_order)
                     + f" | fastest eager baseline | {at_comp} speedup"
                     + f" | fastest compiled baseline | {at_comp} speedup |")
            L.append("|" + "---|" * (7 + len(md_order)))
        else:
            L.append("| Dataset | Molecule | Atoms | " + " | ".join(md_order)
                     + f" | fastest baseline | {at_comp} speedup |")
            L.append("|" + "---|" * (5 + len(md_order)))
        for dataset, order in datasets:
            for mol in order:
                cells = [cell(dataset, m, mol, bs, "latency_median_ms",
                              lambda r: f"{r['latency_median_ms']:.1f}") for m in md_order]
                extra = fastest(dataset, mol, bs, baselines_present)
                if has_comp:
                    extra_c = fastest(dataset, mol, bs, [md_comp_label(m) for m in compiled_present])
                    extra += " | " + extra_c.replace(" (compiled) |", " |")
                L.append(f"| {dataset} | {mol} | {atoms_of(dataset, mol)} | "
                         + " | ".join(cells) + f" | {extra} |")
        L.append("")

    for bs in (1, 4):
        L.append(f"## Peak GPU memory (GB), batch size {bs}")
        L.append("")
        L.append("| Dataset | Molecule | Atoms | " + " | ".join(md_order) + " |")
        L.append("|" + "---|" * (3 + len(md_order)))
        for dataset, order in datasets:
            for mol in order:
                cells = [cell(dataset, m, mol, bs, "peak_mem_gb",
                              lambda r: f"{r['peak_mem_gb']:.2f}") for m in md_order]
                L.append(f"| {dataset} | {mol} | {atoms_of(dataset, mol)} | "
                         + " | ".join(cells) + " |")
        L.append("")

    L.append("## Throughput (structures/s), batch size 4 — median-based (mean-based)")
    L.append("")
    L.append("Median-based = 4000 / median latency (the paper's definition). Mean-based = "
             "4000 / mean latency; it is pulled down by occasional slow calls.")
    L.append("")
    L.append("| Dataset | Molecule | Atoms | " + " | ".join(md_order) + " |")
    L.append("|" + "---|" * (3 + len(md_order)))
    for dataset, order in datasets:
        for mol in order:
            cells = [cell(dataset, m, mol, 4, "throughput_median_samples_per_s",
                          lambda r: f"{r['throughput_median_samples_per_s']:.1f} "
                                    f"({r['throughput_samples_per_s']:.1f})")
                     for m in md_order]
            L.append(f"| {dataset} | {mol} | {atoms_of(dataset, mol)} | "
                     + " | ".join(cells) + " |")
    L.append("")

    # ---- per-dataset parameter footnote ----
    L.append("## Model sizes (parameters, per dataset config)")
    L.append("")
    for m in md_order:
        per = {}
        for r in rows:
            if r["_md"] == m and r["n_params"]:
                per.setdefault(r["dataset"], set()).add(r["n_params"])
        if not per:
            continue
        parts = []
        for ds in ("rmd17", "md22"):
            if ds in per:
                parts.append(f"{'rMD17' if ds == 'rmd17' else 'MD22'}: "
                             + " / ".join(f"{n:,}" for n in sorted(per[ds])))
        L.append(f"- **{m}**: " + "; ".join(parts))
    L.append("")

    # ---- compiled-baseline variants (only when compiled CSVs were read) ----
    if has_comp_rows:
        L.append("## Compiled-baseline variants")
        L.append("")
        L.append("Two torch.compile recipes per baseline (`inductor`: graph break at "
                 "`autograd.grad`, forces from the compiled backward; `fused`: `autograd.grad` "
                 "inlined, forward + force backward in one graph; see Measurement conditions). "
                 "The tables above show the selected variant as `<model> (compiled)`; both are in "
                 f"the long CSV (`selected` column). Selection: `--compiled_variant "
                 f"{args.compiled_variant}`" + (" = lower geometric-mean median latency over the "
                 "rows where both variants are ok (a single variant with ok rows is taken as is)."
                 if args.compiled_variant == "auto" else "."))
        L.append("")
        L.append("| Model | selected | inductor rows ok / oom / compile_error | "
                 "fused rows ok / oom / compile_error | fused / inductor median latency "
                 "(geo-mean over common ok rows) | max compile_graphs_warmup (inductor / fused) | "
                 "rows with compile_graphs_timed > 0 (inductor / fused) |")
        L.append("|---|---|---|---|---|---|---|")
        cerr = []
        for m in baseline_labels:
            if not any(comp_rows[m].values()):
                continue

            def counts(v):
                rs = comp_rows[m][v]
                if not rs:
                    return "— (no file)"
                n_ok = sum(r["status"] == "ok" for r in rs)
                n_oom = sum(r["status"] == "oom" for r in rs)
                n_ce = sum(str(r["status"]).startswith("compile_error") for r in rs)
                return f"{n_ok} / {n_oom} / {n_ce} (of {len(rs)})"

            def maxg(v):
                g = [fint(r["_diag"].get("compile_graphs_warmup")) for r in comp_rows[m][v]]
                g = [x for x in g if x is not None]
                return str(max(g)) if g else "—"

            def timed(v):
                g = [fint(r["_diag"].get("compile_graphs_timed")) for r in comp_rows[m][v]]
                return str(sum(1 for x in g if x)) if comp_rows[m][v] else "—"

            st = variant_stats[m]
            ratio = (f"{st['fused_over_inductor']:.3f} (n = {st['common']})"
                     if st["fused_over_inductor"] else "— (no common ok row)")
            L.append(f"| {m} | {selected.get(m, '— (none)')} | {counts('inductor')} | "
                     f"{counts('fused')} | {ratio} | {maxg('inductor')} / {maxg('fused')} | "
                     f"{timed('inductor')} / {timed('fused')} |")
            for v in COMPILE_VARIANTS:
                for r in comp_rows[m][v]:
                    if str(r["status"]).startswith("compile_error"):
                        cerr.append(f"- {comp_label(m, v)}, {r['dataset']}/{r['molecule']} "
                                    f"bs {r['batch_size']}: `{r['status']}`")
        L.append("")
        if cerr:
            L.append("compile_error rows (no eager fallback; the cell shows `err`):")
            L.append("")
            L.extend(cerr)
            L.append("")

    # ---- fairness notes ----
    L.append("## Measurement conditions (read before comparing columns)")
    L.append("")
    def setting(m, col, texts, default):
        vals = sorted(recorded.get(m, {}).get(col, ()))
        if not vals:
            return default
        return " / ".join(texts.get(v, v) for v in vals)

    synthetic = False
    for m in baselines_present:
        if m in NOTES:
            coords = setting(m, "coords", COORDS_TEXT, "real frames")
            forces = setting(m, "forces", FORCES_TEXT, "autograd forces")
            synthetic |= "synthetic" in recorded.get(m, {}).get("coords", ())
            L.append(f"- **{m}**: {NOTES[m].format(coords=coords, forces=forces)}.")
    if has_comp:
        L.append("- **Compiled baselines** (`<model> (compiled)` columns): " + COMPILED_PROTOCOL
                 + ". The tables show the selected variant ("
                 + ", ".join(f"{m}: {selected[m]}" for m in compiled_present) + ").")
        for m in compiled_present:
            if m in COMPILED_NOTES:
                L.append(f"- **{md_comp_label(m)}**: {COMPILED_NOTES[m]}.")
    elif has_comp_rows:
        L.append("- **Compiled baselines**: measured (" + COMPILED_PROTOCOL + ") but no variant "
                 "produced an ok row" + ("" if args.compiled_variant == "auto" else
                                         f" for the forced `{args.compiled_variant}` recipe")
                 + ", so no compiled column is shown; see \"Compiled-baseline variants\".")
    at_note = args.atomtrick_note
    if at_note is None:
        if tf32_seen == {"off"}:
            at_note = "strict fp32 (--tf32 off)"
        elif tf32_seen == {"on"} or not tf32_seen:
            at_note = ("TF32 matmuls (torch.set_float32_matmul_precision('high'); the baselines "
                       "did NOT use TF32)")
        else:
            at_note = f"mixed tf32 settings {sorted(tf32_seen)}"
    L.append(f"- **{args.atomtrick_label}**: {at_note}; real frames; neighbour list built "
             "outside the timed region; one make_fx graph per batch size traced on the first "
             "molecule and reused (compile time excluded).")
    if synthetic:
        L.append("- Synthetic Gaussian coordinates (σ = 3 Å) give ≈29% of atom pairs within 5 Å; "
                 "the real molecules give 63–100% (rMD17 / small MD22) and ≈11% (nanotube), so "
                 "such graphs are 2–3.5× sparser than real for small systems and ≈2.7× denser "
                 "for the nanotube. Re-run those harnesses with `--coords raw` for a "
                 "like-for-like input.")

    with open(args.out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
