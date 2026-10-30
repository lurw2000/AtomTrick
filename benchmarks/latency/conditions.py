#!/usr/bin/env python3
"""Provenance + consistency report for a run_all.sh results dir (stdlib only).

usage: python benchmarks/latency/conditions.py <OUT> [--gpu_tag h100]

The protocol is the full-fp32 single-session measurement of the reported set
(results/h100_fp32_2026-10-20/, baselines eager and compiled); run_all.sh runs it with the
compiled baselines (raw/<model>_compiled_<inductor|fused>_<dataset>_latency_<tag>.csv), which
are checked and summarised here when present.

Writes <OUT>/MEASUREMENT_CONDITIONS.md and <OUT>/conditions.json, and
<OUT>/latency_all_summary_d5.md = merge.py's summary with its short "Measurement
conditions" section replaced by the conditions actually recorded in the CSVs.
Exit code 1 if a consistency check fails (precision, dtype, force mode, frames, GPU, a
compiled row that compiled new graphs during the timed calls, a compiled recipe whose ok rows
compiled no graph at all).  A compile_error row is a WARNING (listed), not a failed check;
provenance of a compiled recipe is checked on the rows that ran.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("out", help="results dir written by run_all.sh")
_ap.add_argument("--gpu_tag", default="h100", help="file-name tag used by run_all.sh (GPU_TAG)")
_a = _ap.parse_args()
OUT, TAG = _a.out, _a.gpu_tag
BASE = ["equiformer", "equiformer_v2", "gotennet", "mace"]
LABEL = {"equiformer": "Equiformer", "equiformer_v2": "EquiformerV2", "gotennet": "GotenNet",
         "mace": "MACE"}
VARIANTS = ["inductor", "fused"]
problems, warnings, rows_by, comp_by = [], [], {}, {}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def is_cerr(r):
    return str(r.get("status", "")).startswith("compile_error")


# ---------------------------------------------------------------- load
at_path = os.path.join(OUT, "atomtrick_latency.csv")
at = read(at_path) if os.path.exists(at_path) else []
if not at:
    problems.append("ATOMTRICK CSV missing")
for m in BASE:
    for ds in ("rmd17", "md22"):
        p = os.path.join(OUT, "raw", f"{m}_{ds}_latency_{TAG}.csv")
        if os.path.exists(p):
            for r in read(p):
                r["dataset"] = ds
                rows_by.setdefault(m, []).append(r)
        else:
            problems.append(f"missing {p}")
    # compiled baselines: optional as a whole, but a variant with one dataset must have both
    for v in VARIANTS:
        paths = {ds: os.path.join(OUT, "raw", f"{m}_compiled_{v}_{ds}_latency_{TAG}.csv")
                 for ds in ("rmd17", "md22")}
        have = {ds: p for ds, p in paths.items() if os.path.exists(p)}
        if not have:
            continue
        for ds, p in paths.items():
            if ds not in have:
                problems.append(f"missing {p}")
                continue
            for r in read(p):
                r["dataset"] = ds
                comp_by.setdefault((m, v), []).append(r)

envs = {}
for p in sorted(glob.glob(os.path.join(OUT, "env", "*.json"))):
    with open(p) as fh:
        envs[os.path.basename(p)[:-5]] = json.load(fh)

# ---------------------------------------------------------------- checks
gpus = {r["gpu_name"] for r in at if r.get("gpu_name")}
for m, rs in rows_by.items():
    gpus |= {r["gpu_name"] for r in rs if r.get("gpu_name")}
    want = {"matmul_precision": {"highest"}, "cudnn_allow_tf32": {"False"},
            "param_dtype": {"float32"}, "forces": {"autograd"}, "create_graph": {"False"},
            "coords": {"raw"}}
    for k, ok in want.items():
        seen = {r.get(k) for r in rs if r.get("status") not in (None, "") and r.get(k)}
        if not seen or not seen <= ok:
            problems.append(f"{m}: {k} = {sorted(map(str, seen))}, expected {sorted(ok)}")
    bad = [r for r in rs if r["status"] not in ("ok", "oom")]
    if bad:
        problems.append(f"{m}: {len(bad)} rows with status " + "; ".join(
            sorted({r['status'][:60] for r in bad})))
    if "compile" in (rs[0] if rs else {}):          # new harness: eager rows say compile=none
        seen = {r.get("compile") for r in rs}
        if not seen <= {"none", ""}:
            problems.append(f"{m}: eager CSV has compile = {sorted(map(str, seen))}")
def rkey(r):
    return (r["dataset"], r["molecule"], int(r["batch_size"]))


def rname(r):
    return f"{r['dataset']}/{r['molecule']} bs{r['batch_size']}"


eager_lat = {m: {rkey(r): f(r.get("latency_median_ms")) for r in rs if r["status"] == "ok"}
             for m, rs in rows_by.items()}
for (m, v), rs in sorted(comp_by.items()):
    tag = f"{m} compiled ({v})"
    gpus |= {r["gpu_name"] for r in rs if r.get("gpu_name")}
    cerr = [r for r in rs if is_cerr(r)]
    # provenance is checked on the rows that ran (a harness-level setup failure writes bare
    # compile_error rows); compile and pytorch_jit are on every row
    ran = [r for r in rs if not is_cerr(r)]
    want = {"compile": {v}, "pytorch_jit": {"0"}}
    want_ran = {"matmul_precision": {"highest"}, "cudnn_allow_tf32": {"False"},
                "param_dtype": {"float32"}, "forces": {"autograd"},
                "create_graph": {"False"} if v == "inductor" else {"True"}, "coords": {"raw"}}
    for k, ok in want.items():
        seen = {r.get(k) for r in rs if r.get(k)}
        if not seen or not seen <= ok:
            problems.append(f"{tag}: {k} = {sorted(map(str, seen))}, expected {sorted(ok)}")
    for k, ok in want_ran.items():
        if not ran:                     # nothing ran (reported as a WARNING below); a value the
            seen = {r.get(k) for r in rs if r.get(k)}   # bare rows do carry must still be right
            if seen and not seen <= ok:
                problems.append(f"{tag}: {k} = {sorted(map(str, seen))}, expected {sorted(ok)}")
            continue
        seen = {r.get(k) for r in ran if r.get(k)}
        if not seen or not seen <= ok:
            problems.append(f"{tag}: {k} = {sorted(map(str, seen))}, expected {sorted(ok)}")
    if cerr:
        if not ran:
            warnings.append(f"{tag}: ALL {len(cerr)} rows are compile_error (no eager fallback; "
                            f"the recipe never ran): {cerr[0]['status'][:160]}")
        else:
            warnings.append(f"{tag}: {len(cerr)} compile_error rows (no eager fallback): " + "; ".join(
                f"{rname(r)}: {r['status'][:120]}" for r in cerr))
    bad = [r for r in rs if r["status"] not in ("ok", "oom") and not is_cerr(r)]
    if bad:
        problems.append(f"{tag}: {len(bad)} rows with status " + "; ".join(
            sorted({r['status'][:60] for r in bad})))
    okr = [r for r in rs if r["status"] == "ok"]
    timed = [r for r in okr if (f(r.get("compile_graphs_timed")) or 0) > 0]
    if timed:
        problems.append(f"{tag}: {len(timed)} ok rows compiled new graphs during the TIMED calls "
                        "(compile_graphs_timed > 0): " + ", ".join(rname(r) for r in timed))
    # a compiled run must have compiled something: all-zero compile_graphs_warmup over the ok
    # rows means Dynamo never ran (silent eager fallback or torch.compile never reached)
    if okr and sum(int(f(r.get("compile_graphs_warmup")) or 0) for r in okr) == 0:
        problems.append(f"{tag}: no ok row compiled a graph (sum of compile_graphs_warmup = 0): "
                        "the rows were timed eagerly")
    slow = [r for r in okr if (f(r.get("compile_graphs_warmup")) or 0) == 0
            and (f(r.get("compile_retimed")) or 0) == 0
            and (f(r.get("latency_median_ms")) or 0) > (eager_lat.get(m, {}).get(rkey(r)) or 1e30)]
    if slow:
        warnings.append(f"{tag}: {len(slow)} ok rows compiled no new graph and are SLOWER than the "
                        "eager row (possible eager fallback for that shape): "
                        + ", ".join(rname(r) for r in slow))
    nocheck = [r for r in okr if r.get("compile_max_rel_err") in (None, "")]
    if nocheck:
        warnings.append(f"{tag}: {len(nocheck)} ok rows without the compiled-vs-eager output check "
                        "(compile_max_rel_err empty; the eager reference ran out of memory, or an "
                        "older harness): " + ", ".join(rname(r) for r in nocheck))
    if v == "fused":
        broke = [r for r in okr if (f(r.get("compile_graph_breaks")) or 0) > 0]
        if broke:
            warnings.append(f"{tag}: {len(broke)} ok rows with graph breaks during the warm-up "
                            "(compile_graph_breaks > 0): the fused recipe did not give ONE graph "
                            "there (outputs still matched eager): " + ", ".join(rname(r) for r in broke))
    retimed = [r for r in rs if (f(r.get("compile_retimed")) or 0) > 0]
    if retimed:
        warnings.append(f"{tag}: {len(retimed)} rows re-timed after a recompile in the first timed "
                        "loop (compile_retimed=1): " + ", ".join(rname(r) for r in retimed))
for k, ok in {"tf32": {"off"}, "frame_source": {"raw"}}.items():
    seen = {r.get(k) for r in at}
    if at and seen != ok:
        problems.append(f"ATOMTRICK: {k} = {sorted(map(str, seen))}, expected {sorted(ok)}")
bad = [r for r in at if r["status"] not in ("ok", "oom")]
if bad:
    problems.append(f"ATOMTRICK: {len(bad)} rows with status " + "; ".join(
        sorted({r['status'][:60] for r in bad})))
if len(gpus) != 1:
    problems.append(f"GPU names differ across models: {sorted(gpus)}")

meta = open(os.path.join(OUT, "run_meta.txt")).read() if os.path.exists(
    os.path.join(OUT, "run_meta.txt")) else ""
jobs = sorted({ln.split()[0].split("=")[1] for ln in meta.splitlines()
               if ln.startswith("slurm_job_id=")})
# GPU UUIDs from the nvidia-smi rows only (the driver's own lines are key=value, and its "env:"
# line may hold a GPU-... UUID in CUDA_VISIBLE_DEVICES); run_all.sh queries only the timed GPU
uuids = sorted({tok for ln in meta.splitlines() if "=" not in ln
                for tok in ln.replace(",", " ").split() if tok.startswith("GPU-")})
if len(jobs) > 1 or len(uuids) > 1:
    problems.append(f"results come from {len(jobs)} jobs / {len(uuids)} GPUs ({jobs}, {uuids}): "
                    "not one session -- re-run with FORCE=1")

# ---------------------------------------------------------------- tables
def key(r):
    return (r["dataset"], r["molecule"], int(r["batch_size"]))


at_edges = {key(r): r.get("edges") for r in at if r["path"] == "eager"}
order = []
for r in at:
    if r["path"] == "eager" and key(r) not in order:
        order.append(key(r))
edges = {m: {key(r): r.get("edges") for r in rs} for m, rs in rows_by.items()}
gb = {m: {key(r): f(r.get("graph_build_median_ms")) for r in rs} for m, rs in rows_by.items()}
lat = {m: {key(r): f(r.get("latency_median_ms")) for r in rs if r["status"] == "ok"}
       for m, rs in rows_by.items()}
atoms = {key(r): r.get("atoms") for r in at}
has_comp = bool(comp_by)

L = ["# Latency measurement conditions (recorded, not assumed)", ""]
L.append(f"Results dir: `{OUT}`")
L.append("")
L.append("## Common protocol")
L.append("")
L.append("- One job, one GPU (" + ", ".join(sorted(gpus)) + f"); SLURM job(s) {', '.join(jobs) or '?'}; "
         f"GPU UUID(s) {', '.join(uuids) or '?'}. Models run sequentially; driver.log has a WARNING line if any other process was on the GPU.")
L.append("- Strict fp32 for every model: `torch.set_float32_matmul_precision('highest')`, fp32 "
         "weights and inputs (MACE explicitly `--dtype float32`; MACE-OFF/MACE papers train in "
         "float64). cuDNN TF32 off for the baselines; bench_atomtrick.py leaves "
         "`torch.backends.cudnn.allow_tf32` at the PyTorch default (True), which affects only "
         "cuDNN convolutions, and ATOMTRICK has none. Random-init weights (latency does not depend "
         "on weight values).")
L.append("- Inputs: frames 0..bs-1 of the raw rMD17 / MD22 trajectories for every model (real "
         "coordinates; no synthetic Gaussian batches).")
L.append("- One call = energy + forces F = -dE/dR by autograd with `create_graph=False` for every "
         "model (Equiformer: upstream hard-codes True, rewritten by `--create-graph false`; "
         "GotenNet: output-head attribute; EquiformerV2: built without its direct-force head, "
         "`--forces autograd`; MACE `training=False`; ATOMTRICK eager in eval mode). ATOMTRICK "
         "compiled: one make_fx graph per batch size (forward + force backward, traced with "
         "train_mode=True, the only trace make_fx captures correctly) compiled with "
         "torch.compile(dynamic=True) and executed under `torch.no_grad()`, so no second-order "
         "graph is built at run time."
         + (" Compiled baselines, `fused` recipe: `create_graph=True` inside the compiled graph "
            "(the traced `autograd.grad` must be differentiable), as recorded in their CSVs; "
            "`inductor` recipe: `create_graph=False` as eager." if has_comp else ""))
L.append("- 10 warm-up calls (absorb make_fx trace + Inductor compile for ATOMTRICK compiled"
         + (", and Dynamo trace + Inductor compile for the compiled baselines; see "
            "`compile_warmup_s` below" if has_comp else "")
         + ") + 50 calls, each bracketed by CUDA events; latency = median; peak memory = "
         "`max_memory_allocated` over the timed calls (includes weights).")
if has_comp:
    mace_env = envs.get("mace_compiled", {})
    if any(m == "mace" for (m, v) in comp_by):
        py = mace_env.get("python")
        mace_txt = ("; MACE compiled runs in its own environment (Python "
                    + (str(py) if py else "<= 3.12, see the Environments table")
                    + "; torch 2.5.1 has no torch.compile on Python 3.13, the interpreter of "
                    "the original MACE environment)")
    else:
        mace_txt = ""
    compiled_txt = ("baselines compiled with two recipes each (same harness, inputs and precision as "
                    "eager; whole energy + force callable under `torch.compile(dynamic=True)`, "
                    "TorchInductor default mode, no CUDA graphs; parameters frozen with "
                    "`requires_grad_(False)`; `PYTORCH_JIT=0` before `import torch`; e3nn "
                    "TorchScript codegen off; Dynamo never reset between molecules; MACE and "
                    "GotenNet compiled reuse one random-init model instance for all molecules so "
                    "the dynamic-shape graph is shared): `inductor` = graph break at "
                    "`torch.autograd.grad`, energy forward compiled, forces from AOTAutograd's "
                    "compiled backward; `fused` = `torch.autograd.grad` allowed in graph, forward + "
                    "force backward in ONE compiled graph (MACE's upstream torch.compile recipe, "
                    "generalised), traced with grad enabled, outputs detached inside the compiled "
                    "callable so AOTAutograd compiles an inference graph executed under no_grad "
                    "(the shape of ATOMTRICK's make_fx step); after the warm-up every compiled row's "
                    "energy and forces were compared with one eager call of the same model "
                    "(`compile_max_rel_err`, mismatch -> compile_error); the PyG models still build "
                    "their radius graph inside the timed forward when compiled" + mace_txt)
else:
    compiled_txt = "only ATOMTRICK is compiled"
L.append("- Not equalised (architectural/harness facts, disclosed): neighbour list built inside the "
         "timed forward for Equiformer, EquiformerV2 and GotenNet (cost measured separately below), "
         "outside for ATOMTRICK (CPU cKDTree) and MACE; neighbour caps: Equiformer 1000, "
         "EquiformerV2 50, GotenNet 32 (+ self loops), ATOMTRICK/MACE none; EquiformerV2 = OC20 "
         "N@12_L@6_M@2 architecture (never trained on these data); GotenNet per-dataset configs; "
         f"{compiled_txt}; each baseline runs in its own environment (torch versions "
         "below).")
L.append("")
L.append("## Environments")
L.append("")
L.append("| model | python | torch | CUDA (torch) | key packages | matmul precision while timing |")
L.append("|---|---|---|---|---|---|")
for name, e in sorted(envs.items()):
    if name == "atomtrick":
        prec = "highest (bench --tf32 off)"
    elif name.endswith("_compiled"):
        prec = ", ".join(sorted({r.get("matmul_precision") or "" for (m, v), rs in comp_by.items()
                                 if m == name[:-len("_compiled")] for r in rs} - {""})) or "—"
    else:
        prec = ", ".join(sorted({r.get("matmul_precision") or "" for r in rows_by.get(name, [])}
                                - {""})) or "—"
    mods = ", ".join(f"{k} {v}" for k, v in e.get("modules", {}).items() if k != "mace_file")
    L.append(f"| {name} | {e.get('python')} | {e.get('torch')} | {e.get('torch_cuda')} | {mods} | {prec} |")
L.append("")
L.append("## Graph sizes (directed edges in the timed graph) and in-forward graph-build cost")
L.append("")
L.append("| dataset | molecule | atoms | bs | ATOMTRICK | MACE | Equiformer | EquiformerV2 (cap 50) | "
         "GotenNet (cap 32, +loops) | graph build ms: Eqf / EqV2 / GotenNet |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for k in order:
    def e(m):
        return edges.get(m, {}).get(k) or "—"

    def g(m):
        v = gb.get(m, {}).get(k)
        return f"{v:.2f}" if v is not None else "—"
    L.append(f"| {k[0]} | {k[1]} | {atoms.get(k)} | {k[2]} | {at_edges.get(k) or '—'} | {e('mace')} | "
             f"{e('equiformer')} | {e('equiformer_v2')} | {e('gotennet')} | "
             f"{g('equiformer')} / {g('equiformer_v2')} / {g('gotennet')} |")
L.append("")


def geo_stats(a, b):
    """geometric mean, range and count of a[k]/b[k] over the common keys (None if none)."""
    common = sorted(set(a) & set(b))
    rs = [a[k] / b[k] for k in common if a[k] and b[k]]
    if not rs:
        return None
    return statistics.geometric_mean(rs), min(rs), max(rs), len(rs)


def ratio_block(title, a, b, la, lb):
    st = geo_stats(a, b)
    if st is None:
        return
    gm, lo, hi, n = st
    L.append(f"## {title}")
    L.append("")
    L.append(f"{la} / {lb}, median latency: geometric-mean "
             f"{gm:.3f}, range {lo:.3f}-{hi:.3f} "
             f"over {n} (molecule, bs) rows.")
    L.append("")


# ---------------------------------------------------------------- compiled baselines
if has_comp:
    L.append("## Compiled baselines")
    L.append("")
    L.append("Per model and torch.compile recipe (`inductor`: graph break at `autograd.grad`; "
             "`fused`: `autograd.grad` inlined in one graph). compile_warmup_s = wall-clock seconds "
             "of the 10 warm-up calls (trace + compile, excluded from the latency); "
             "compile_graphs_warmup = unique graphs compiled during a row's warm-up; "
             "compile_graph_breaks = graph-break count delta during the warm-up (sum over rows, "
             "median per row); max rel. err = largest `compile_max_rel_err` (compiled vs eager "
             "energy / forces after the warm-up, max relative error; the harness fails the row "
             "above 1e-2); compiled / eager = geometric-mean ratio of the median latency over "
             "the (dataset, molecule, bs) rows ok in both. A row that compiled a new graph during "
             "the timed calls (compile_graphs_timed > 0), or a recipe whose ok rows compiled no "
             "graph at all, is a failed check.")
    L.append("")
    L.append("| model | recipe | rows ok / oom / compile_error | compile_warmup_s median / max | "
             "max compile_graphs_warmup | compile_graph_breaks sum (median per row) | "
             "rows compile_graphs_timed > 0 | rows re-timed | max rel. err vs eager | "
             "compiled / eager latency (n) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for (m, v), rs in sorted(comp_by.items()):
        n_ok = sum(r["status"] == "ok" for r in rs)
        n_oom = sum(r["status"] == "oom" for r in rs)
        n_ce = sum(is_cerr(r) for r in rs)
        ws = [f(r.get("compile_warmup_s")) for r in rs]
        ws = [x for x in ws if x is not None]
        gw = [f(r.get("compile_graphs_warmup")) for r in rs]
        gw = [int(x) for x in gw if x is not None]
        br = [f(r.get("compile_graph_breaks")) for r in rs]
        br = [int(x) for x in br if x is not None]
        n_timed = sum(1 for r in rs if (f(r.get("compile_graphs_timed")) or 0) > 0)
        n_ret = sum(1 for r in rs if (f(r.get("compile_retimed")) or 0) > 0)
        re_ = [f(r.get("compile_max_rel_err")) for r in rs if r["status"] == "ok"]
        re_ = [x for x in re_ if x is not None]
        clat = {key(r): f(r.get("latency_median_ms")) for r in rs if r["status"] == "ok"}
        st = geo_stats(clat, lat.get(m, {}))
        L.append(f"| {LABEL[m]} | {v} | {n_ok} / {n_oom} / {n_ce} (of {len(rs)}) | "
                 + (f"{statistics.median(ws):.1f} / {max(ws):.1f}" if ws else "—") + " | "
                 + (str(max(gw)) if gw else "—") + " | "
                 + (f"{sum(br)} ({statistics.median(br):g})" if br else "—")
                 + f" | {n_timed} | {n_ret} | "
                 + (f"{max(re_):.1e} ({len(re_)} rows)" if re_ else "—") + " | "
                 + (f"{st[0]:.3f} ({st[3]})" if st else "—") + " |")
    L.append("")
    for m in BASE:
        a = {key(r): f(r.get("latency_median_ms")) for r in comp_by.get((m, "fused"), [])
             if r["status"] == "ok"}
        b = {key(r): f(r.get("latency_median_ms")) for r in comp_by.get((m, "inductor"), [])
             if r["status"] == "ok"}
        ratio_block(f"{LABEL[m]}: fused vs inductor recipe", a, b, "fused", "inductor")


def at_lat(path_csv, path):
    if not os.path.exists(path_csv):
        return {}
    return {key(r): f(r["latency_median_ms"]) for r in read(path_csv)
            if r["path"] == path and r["status"] == "ok"}


eul = os.path.join(OUT, "extra", "atomtrick_euler_latency.csv")
for p in ("eager", "makefx"):
    ratio_block(f"Extra: ATOMTRICK pole-free vs Euler edge frame ({p})",
                at_lat(at_path, p), at_lat(eul, p), "polefree (default)", "euler (direct Euler-angle evaluation)")

L.append("## Consistency checks")
L.append("")
L.extend([f"- FAIL: {p}" for p in problems] or ["- all checks passed (precision, dtype, force "
                                               "mode, create_graph, frames, single GPU"
                                               + (", compiled rows" if has_comp else "") + ")"])
L.extend([f"- WARNING: {w}" for w in warnings])
L.append("")
md = "\n".join(L) + "\n"
with open(os.path.join(OUT, "MEASUREMENT_CONDITIONS.md"), "w") as fh:
    fh.write(md)
with open(os.path.join(OUT, "conditions.json"), "w") as fh:
    json.dump({"problems": problems, "warnings": warnings, "gpus": sorted(gpus), "jobs": jobs,
               "gpu_uuids": uuids, "envs": envs}, fh, indent=1)

summ = os.path.join(OUT, "latency_all_summary.md")
if os.path.exists(summ):
    s = open(summ).read()
    cut = s.find("## Measurement conditions")
    head = s[:cut] if cut >= 0 else s
    with open(os.path.join(OUT, "latency_all_summary_d5.md"), "w") as fh:
        fh.write(head + md.split("\n", 2)[2].replace("## Common protocol",
                                                    "## Measurement conditions (recorded)"))
print(md)
sys.exit(1 if problems else 0)
