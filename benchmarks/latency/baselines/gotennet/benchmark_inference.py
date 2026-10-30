#!/usr/bin/env python3
"""Measure GotenNet inference latency on MD22 / rMD17.

For each molecule and batch size, instantiates the model (fresh random
weights — latency depends on architecture + input shape, not trained params),
loads a batch from the cached dataset, runs warmup iterations, then times
many forward passes end-to-end with torch.cuda.Event.

GotenNet computes forces as -dE/dx via autograd with create_graph=True
(MDTask: derivative="force"; switched off by the default --create-graph
false). So forward() internally enables grad; we must NOT wrap timing in
torch.no_grad() or inference_mode(). model.eval() is the right switch: it
disables dropout without touching autograd.

Each dataset uses its own production experiment config:
  md22  -> experiment=md22_at_at   (n_interactions=8,  n_atom_basis=384, lmax=2)
  rmd17 -> experiment=rmd17_aspirin (n_interactions=12, n_atom_basis=192, lmax=2)

Output: results/gotennet_<dataset>_latency.csv with per-(molecule, batch_size)
rows including mean/median/p10/p90 ms, throughput, peak GPU memory, and a
status flag ('ok'/'oom'/'error') for each configuration.

Usage:
    python scripts/benchmark_inference.py                            # md22, bs 1 & 4
    python scripts/benchmark_inference.py --dataset rmd17 --batch-sizes 1 4
    python scripts/benchmark_inference.py --molecules AT_AT DHA      # subset

--- ATOMTRICK release notes (benchmarks/latency/baselines/gotennet/) ---------------------
Harness for the GotenNet rows of the latency tables.  --repo-root / --data-dir are required
arguments (or env vars), and the switches below select the force protocol.  The DEFAULTS
reproduce the reported numbers.
  * --repo-root : a clone of sarpaykent/GotenNet at 16c7026 (branch rMD17) with
    gotennet_local.patch applied (adds the MD22 dataset/experiment config and changes the
    rmd17_aspirin experiment to n_interactions=12).  Install it (pip install -e) or put it on
    PYTHONPATH.
  * Configs: rMD17 -> experiment=rmd17_aspirin (12 layers, 192 ch, 21,069,057 params);
             MD22  -> experiment=md22_at_at   ( 8 layers, 384 ch, 42,603,777 params).
  * --data-dir : root holding raw/rmd17/... and raw/md22/md22_<mol>.npz (GotenNet's own
    loaders; it processes to .pt on first use).  Real frames 0..bs-1 of the full trajectory.
  * --create-graph {true,false} (default false = reported).  false sets create_graph=False on
    the AtomwiseV3 output head (its own constructor attribute), i.e. the inference force path
    with no second-order graph; true = upstream MDTask.
  * --matmul-precision {highest,high,medium} (default highest = strict fp32), set before timing.
  * Extra CSV columns: edges (the model's own radius graph: r=cutoff, loop=True,
    max_num_neighbors=32), graph_build_median_ms (that call timed alone), provenance.
  * radius_graph is built INSIDE the model (timed).  Eager PyTorch, strict fp32 (no TF32).

--- --compile {none,inductor,fused} (default none = the eager rows above, unchanged) ----------
  Times the SAME model under torch.compile(dynamic=True) (TorchInductor, default mode, no CUDA
  graphs).  The compiled object is the whole model call (radius graph + energy + the AtomwiseV3
  head's force autograd.grad):
    inductor : head create_graph=False (as the eager rows); Dynamo graph-breaks at
               autograd.grad and AOTAutograd's compiled backward computes the forces.
    fused    : torch._dynamo.allow_in_graph(torch.autograd.grad) with head create_graph=True
               (the upstream MDTask value), so the force backward is inlined into ONE compiled
               graph (MACE's upstream recipe, generalised); the outputs are detached inside the
               compiled callable (lc.fused_callable) so AOTAutograd compiles an inference graph
               run under no_grad, like ATOMTRICK's make_fx step.  create_graph=True is recorded.
               The radius graph (torch_cluster, a custom op without a fake kernel) is built by a
               nested module, so the graph break it causes lands at the representation call and
               the inlined autograd.grad may not reach batch.pos -> such rows are compile_error.
  --create-graph is ignored with --compile (the recipe fixes it).  Both: PYTORCH_JIT=0 before
  import torch, parameters frozen (requires_grad_(False)), batch.pos.requires_grad_(True) set
  by the harness, 10 warmup (trace + compile, excluded) + 50 timed calls as for eager rows;
  after the warmup the compiled energy + forces are compared with one eager call of the same
  model (compile_max_rel_err; a mismatch is a compile_error).  The compiled model is built ONCE
  per run (latency is weight-agnostic) so the dynamic-shape graph is reused across molecules;
  eager rows keep one fresh model per molecule.  Extra CSV columns: compile,
  compile_graphs_warmup, compile_graphs_timed, compile_graph_breaks, compile_warmup_s,
  compile_retimed, compile_max_rel_err, pytorch_jit.  A compile failure is recorded as status
  "compile_error: ..." (no eager fallback).
  The GotenNet clone must carry gotennet_compile.patch (this directory): the representation's
  in-place `h.unsqueeze_(1)` and two boolean-mask in-place assignments (edge_vec / edge_weight
  of self-loop edges) rewritten out of place.  Same values (checked: eager outputs of the
  original and the patched clone agree to ~1e-6 relative); the in-place forms make
  torch.compile fail ("modified by an inplace operation" on a tensor a compiled fragment saved
  for its backward).  The eager rows run the patched clone too.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import latency_common as lc  # noqa: E402  (baselines/latency_common.py; no torch import)
lc.pre_import_env(sys.argv)   # PYTORCH_JIT=0 for --compile, must precede `import torch`

import argparse  # noqa: E402
import csv  # noqa: E402
import statistics  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

# gotennet is imported in main() so that --repo-root can be put on sys.path first.
MD22 = rMD17 = None


# (molecule, atoms) — ordered by #atoms so the CSV shows latency growth cleanly
MD22_MOLECULES = [
    ("Ac_Ala3_NHMe",          42),
    ("DHA",                   56),
    ("AT_AT",                 60),
    ("stachyose",             87),
    ("AT_AT_CG_CG",          118),
    ("buckyball_catcher",    148),
    ("double_walled_nanotube", 370),
]

RMD17_MOLECULES = [
    ("ethanol", 9), ("malonaldehyde", 9), ("benzene", 12), ("uracil", 12),
    ("toluene", 15), ("salicylic", 16), ("naphthalene", 18), ("paracetamol", 20),
    ("aspirin", 21), ("azobenzene", 24),
]

DATASETS = {"md22": MD22_MOLECULES, "rmd17": RMD17_MOLECULES}
EXPERIMENTS = {"md22": "md22_at_at", "rmd17": "rmd17_aspirin"}


def build_cfg(molecule: str, data_dir: str, config_dir: str, dataset: str = "md22"):
    """Compose the per-dataset experiment config with per-molecule overrides."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={EXPERIMENTS[dataset]}",
                f"datamodule.hparams.dataset_arg={molecule}",
                f"label={molecule}",
                f"paths.data_dir={data_dir}",
            ],
        )
    return cfg


def load_batch(molecule: str, batch_size: int, data_dir: str, device,
               dataset: str = "md22") -> Batch:
    ds = (MD22(root=data_dir, dataset_arg=molecule) if dataset == "md22"
          else rMD17(root=data_dir, dataset_arg=molecule))
    n = min(batch_size, len(ds))
    data_list = [ds[i] for i in range(n)]
    # If the requested batch is larger than the dataset, pad by repetition
    # so the measured batch matches the requested size.
    while len(data_list) < batch_size:
        data_list.append(ds[len(data_list) % len(ds)])
    return Batch.from_data_list(data_list).to(device)


def timed_forwards(model, batch, warmup: int, iters: int):
    """Warm up, then time `iters` forward passes with cuda events.

    Returns (times_ms_list, peak_mem_bytes). The forward enables autograd
    internally and computes forces as grad(E, pos), so this is a full
    inference (energy + forces) latency measurement.
    """
    # Warmup
    for _ in range(warmup):
        _ = model(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _ = model(batch)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    peak_bytes = torch.cuda.max_memory_allocated()
    return times_ms, peak_bytes


def summarize(times_ms, batch_size):
    srt = sorted(times_ms)
    n = len(srt)
    return {
        "latency_mean_ms":        statistics.mean(times_ms),
        "latency_median_ms":      statistics.median(times_ms),
        "latency_p10_ms":         srt[max(0, int(0.1 * n) - 1)],
        "latency_p90_ms":         srt[min(n - 1, int(0.9 * n))],
        "latency_std_ms":         statistics.stdev(times_ms) if n > 1 else 0.0,
        "throughput_samples_per_s": batch_size * 1000.0 / statistics.mean(times_ms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.environ.get("GOTENNET_ROOT"),
                    help="GotenNet clone (16c7026 + gotennet_local.patch); env GOTENNET_ROOT")
    ap.add_argument("--data-dir",  default=os.environ.get("GOTENNET_DATA_DIR"),
                    help="dataset root with raw/rmd17 and raw/md22; env GOTENNET_DATA_DIR")
    ap.add_argument("--output",    default="results/gotennet_md22_latency.csv")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4],
                    help="default 1 4 (reported); the original harness defaulted to 4 128")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters",  type=int, default=50)
    ap.add_argument("--molecules", nargs="+", default=None,
                    help="Subset of molecules; default = all in the dataset")
    ap.add_argument("--dataset", default="md22", choices=["md22", "rmd17"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--create-graph", choices=["true", "false"], default="false",
                    help="false = inference setting, no second-order graph (reported); true = "
                         "upstream create_graph=True force head; ignored "
                         "with --compile (inductor -> false, fused -> true)")
    lc.add_common_args(ap)
    args = ap.parse_args()
    fused = args.compile == "fused"
    create_graph = fused if args.compile != "none" else (args.create_graph == "true")
    if not args.repo_root or not args.data_dir:
        ap.error("--repo-root and --data-dir are required (or env GOTENNET_ROOT / "
                 "GOTENNET_DATA_DIR)")
    args.repo_root = str(Path(args.repo_root).resolve())
    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)
    global MD22, rMD17
    from gotennet.datamodules.components.md22 import MD22
    from gotennet.datamodules.components.rmd17 import rMD17

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for a GPU latency benchmark", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB)", flush=True)

    config_dir = str(Path(args.repo_root) / "gotennet" / "configs")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    base = DATASETS[args.dataset]
    mols = (
        base if args.molecules is None
        else [(m, dict(base)[m]) for m in args.molecules]
    )

    fieldnames = [
        "molecule", "atoms", "batch_size", "total_atoms_per_batch", "n_params",
        "warmup", "iters",
        "latency_mean_ms", "latency_median_ms", "latency_p10_ms",
        "latency_p90_ms", "latency_std_ms",
        "throughput_samples_per_s", "peak_mem_gb",
        "gpu_name", "status",
    ] + lc.PROV_FIELDS

    # Open CSV up-front and flush after every row so partial results survive
    # wall-time kills, node failures, and uncaught exceptions anywhere below.
    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    def emit(row):
        writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())

    prov = dict(lc.compile_prov(args.compile))   # every row carries compile / pytorch_jit

    def blank_row(mol, n_atoms, bs, n_params=None, status="pending"):
        return {
            "molecule": mol, "atoms": n_atoms,
            "batch_size": bs, "total_atoms_per_batch": bs * n_atoms,
            "n_params": n_params,
            "warmup": args.warmup, "iters": args.iters,
            "gpu_name": gpu_name,
            "status": status,
            "latency_mean_ms": None, "latency_median_ms": None,
            "latency_p10_ms": None, "latency_p90_ms": None,
            "latency_std_ms": None, "throughput_samples_per_s": None,
            "peak_mem_gb": None,
            **prov,
        }

    # --compile: the model is built once (first molecule) and the compiled object reused for
    # the whole run, so the dynamic-shape graph is shared across molecules.
    compiled = cstats = None
    compile_setup_error = None      # a failure here is a compile failure of every row
    if args.compile != "none":
        try:
            print(f"compile={args.compile}: PYTORCH_JIT={lc.pytorch_jit_state()} "
                  f"dynamo={lc.configure_dynamo()}", flush=True)
            print(f"  autograd.grad allowed in graph: {lc.set_autograd_in_graph(fused)}", flush=True)
            cstats = lc.CompileStats()
        except Exception as e:
            print(f"COMPILE SETUP FAILED: {e}", flush=True)
            traceback.print_exc()
            compile_setup_error = lc.short_msg(e)

    try:
        cuda_dead = None   # a sticky CUDA error (illegal memory access, device-side assert) kills the
        #                  # context: the remaining rows are recorded as not run instead of crashing the file
        for mol, n_atoms in mols:
            print(f"\n=== {mol} ({n_atoms} atoms) ===", flush=True)
            if compile_setup_error is not None:
                for bs in args.batch_sizes:
                    emit(blank_row(mol, n_atoms, bs, status=f"compile_error: {compile_setup_error}"))
                continue

            # Guard config build + model instantiate + dataset load at the
            # per-molecule level. First-access on uncached molecules can be
            # slow (processing .npz → .pt), and we want one bad molecule not
            # to kill results for the others.
            try:
                if args.compile == "none" or compiled is None:
                    cfg = build_cfg(mol, args.data_dir, config_dir, args.dataset)
                    # The real training pipeline passes dataset_meta=datamodule.get_metadata(label)
                    # for mean/std normalization of energy. We bypass the DataModule, so inject
                    # placeholder stats — they don't affect forward-pass wall-time (latency is
                    # architecture-bound, not weight-bound).
                    dataset_meta = {"mean": 0.0, "std": 1.0}
                    model = instantiate(cfg.model, dataset_meta=dataset_meta).to(device).eval()
                    n_params = sum(p.numel() for p in model.parameters())
                    heads = [m for m in model.modules()
                             if hasattr(m, "create_graph") and hasattr(m, "derivative")]
                    if not heads:
                        raise RuntimeError("no output head with a create_graph attribute found")
                    for m in heads:
                        m.create_graph = create_graph
                    nbr = [m for m in model.modules()
                           if hasattr(m, "max_num_neighbors") and hasattr(m, "cutoff") and hasattr(m, "loop")]
                    nbr = nbr[0] if nbr else None
                    if args.compile != "none":
                        lc.freeze_params(model)
                        compiled = lc.compile_callable(lc.fused_callable(model) if fused else model)
                prov = dict(lc.set_precision(args.matmul_precision),
                            param_dtype=lc.param_dtype(model), forces="autograd",
                            create_graph=all(m.create_graph for m in heads), coords="raw",
                            **lc.compile_prov(args.compile))
                print(f"  precision: {prov}  radius graph: r={getattr(nbr, 'cutoff', '?')} "
                      f"cap={getattr(nbr, 'max_num_neighbors', '?')} "
                      f"loop={getattr(nbr, 'loop', '?')}", flush=True)
            except Exception as e:
                print(f"  SETUP FAILED: {e}", flush=True)
                traceback.print_exc()
                for bs in args.batch_sizes:
                    emit(blank_row(mol, n_atoms, bs, status=f"setup_error: {str(e)[:100]}"))
                continue

            for bs in args.batch_sizes:
                row = blank_row(mol, n_atoms, bs, n_params=n_params, status="ok")
                if cuda_dead is not None:
                    row["status"] = (("compile_error: " if compiled is not None else "error: ")
                                     + f"not run, CUDA context lost earlier ({cuda_dead})")
                    emit(row)
                    continue
                try:
                    batch = load_batch(mol, bs, args.data_dir, device, args.dataset)
                    if nbr is not None:
                        from torch_cluster import radius_graph
                        rg = (lambda: radius_graph(batch.pos, r=float(nbr.cutoff), batch=batch.batch,
                                                   loop=bool(getattr(nbr, "loop", True)),
                                                   max_num_neighbors=int(nbr.max_num_neighbors)))
                        row["edges"] = int(rg().shape[1])
                        row["graph_build_median_ms"] = lc.median_cuda_ms(rg)
                    if args.compile != "none" and compiled is None:
                        raise RuntimeError("compile setup produced no compiled callable")
                    if compiled is None:
                        times_ms, peak_bytes = timed_forwards(
                            model, batch, args.warmup, args.iters,
                        )
                    else:
                        batch.pos.requires_grad_(True)   # the model's _enable_grads, done outside the graph

                        def call():
                            # forward() leaves grad mode DISABLED on exit; enter the compiled
                            # frame with grad enabled every time (same Dynamo guard state, and
                            # the fused recipe's inlined autograd.grad needs grad mode)
                            with torch.enable_grad():
                                return compiled(batch)
                        prop, deriv = heads[0].property, heads[0].derivative
                        times_ms, peak_bytes, cinfo = lc.compiled_timed_forwards(
                            call, args.warmup, args.iters, cstats,
                            # output check: the same (un-compiled) model, same batch and heads
                            eager_call=lambda: model(batch),
                            pick=lambda o: (o[prop], o[deriv]),
                        )
                        row.update(cinfo)
                    row.update(summarize(times_ms, bs))
                    row["peak_mem_gb"] = peak_bytes / (1024**3)
                    print(
                        f"  bs={bs:>3}: median={row['latency_median_ms']:7.2f} ms  "
                        f"mean={row['latency_mean_ms']:7.2f} ms  "
                        f"p10={row['latency_p10_ms']:7.2f}  p90={row['latency_p90_ms']:7.2f}  "
                        f"thr={row['throughput_samples_per_s']:8.1f} samples/s  "
                        f"peak={row['peak_mem_gb']:5.2f} GB",
                        flush=True,
                    )
                except RuntimeError as e:
                    msg = str(e)
                    if lc.cuda_fatal(msg):
                        cuda_dead = lc.short_msg(e, 60)
                        print(f"  bs={bs:>3}: FATAL CUDA ERROR (remaining rows not run): {msg[:200]}", flush=True)
                        traceback.print_exc()
                        row["status"] = (f"compile_error: {lc.short_msg(e)}" if compiled is not None
                                         else f"error: {msg[:100]}")
                    elif "out of memory" in msg.lower():
                        print(f"  bs={bs:>3}: OOM", flush=True)
                        row["status"] = "oom"
                        torch.cuda.empty_cache()
                    elif compiled is not None:
                        print(f"  bs={bs:>3}: COMPILE ERROR: {msg[:200]}", flush=True)
                        traceback.print_exc()
                        row["status"] = f"compile_error: {lc.short_msg(e)}"
                    else:
                        print(f"  bs={bs:>3}: ERROR: {msg[:200]}", flush=True)
                        traceback.print_exc()
                        row["status"] = f"error: {msg[:100]}"
                except Exception as e:
                    print(f"  bs={bs:>3}: ERROR: {e}", flush=True)
                    traceback.print_exc()
                    row["status"] = (f"compile_error: {lc.short_msg(e)}" if compiled is not None
                                     else f"error: {str(e)[:100]}")
                emit(row)
                if cstats is not None:
                    cstats.report_breaks_once()

            if compiled is None:
                del model
            if cuda_dead is None:   # every CUDA call fails on a lost context
                torch.cuda.empty_cache()
    finally:
        csv_file.close()
        print(f"\nCSV saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
