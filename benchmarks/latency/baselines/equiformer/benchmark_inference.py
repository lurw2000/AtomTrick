#!/usr/bin/env python3
"""Measure Equiformer (v1) inference latency on MD22.

For each MD22 molecule and batch size, instantiates the model (fresh random
weights -- latency depends on architecture + input shape, not trained params),
creates a batch (raw-trajectory frames by default; see --coords), runs warmup
iterations, then times many forward passes end-to-end with torch.cuda.Event.

Equiformer computes forces as -dE/dx via autograd with create_graph=True
(rewritten to False by the default --create-graph false), so forward()
internally enables grad; we must NOT wrap timing in torch.no_grad() or
inference_mode(). model.eval() is the right switch.

Uses the default MD17 "nonlinear_exp_l2" configuration from the paper:
  - 6 layers, lmax=2, 4 attention heads
  - irreps_node_embedding: 128x0e+64x1e+32x2e
  - basis_type: exp, number_of_basis: 32
  - ~3.5M parameters

Output: results/equiformer_md22_latency.csv

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --molecules AT_AT DHA
    python scripts/benchmark_inference.py --batch-sizes 1 4 16

--- ATOMTRICK release notes (benchmarks/latency/baselines/equiformer/) -------------------
Harness for the Equiformer rows of the latency tables.  Paths are CLI arguments, and the
switches below select the inputs and the force protocol.  The DEFAULTS reproduce the reported
numbers.
  * --repo-root   : a clone of atomicarchitects/equiformer @ 64cb786 with nets_init.patch
                    applied and the ocpmodels shim (../equiformer_v2/ocpmodels_shim) importable.
  * --coords raw --raw-root R (DEFAULT, = reported numbers): frames 0..bs-1 of the raw rMD17 /
    MD22 downloads, the same inputs as the MACE / GotenNet harnesses and
    bench_atomtrick.py --frame_source raw.
  * --coords synthetic: pos ~ N(0, 3 A), z ~ U{1..9}.  The resulting radius graph is 2-3.5x
    SPARSER than the real rMD17 / small-MD22 molecules and ~2.7x DENSER than the real nanotube,
    so it is not a like-for-like input; not used for any reported row.
  * --coords real --data-root data : the first bs frames of the ATOMTRICK processed test split
    (<data-root>/<dataset>/<mol>_test_default.npz), i.e. the same structures ATOMTRICK is timed on.
  * --create-graph {true,false} (default false = reported).  false forces create_graph=False in
    the model's force autograd.grad call (upstream hard-codes True).
  * --matmul-precision {highest,high,medium} (default highest = strict fp32), set before timing.
  * Extra CSV columns: edges (radius graph the model builds, same r / neighbour cap),
    graph_build_median_ms (that radius_graph call timed alone), precision / dtype / force /
    version provenance.
  * The radius graph is built INSIDE model.forward (timed); ATOMTRICK and MACE build theirs
    outside the timed region.  Eager PyTorch, strict fp32 (no TF32), random-init weights.

--- --compile {none,inductor,fused} (default none = the eager rows above, unchanged) ----------
  Times the SAME model under torch.compile(dynamic=True) (TorchInductor, default mode, no CUDA
  graphs).  The compiled object is the whole model.forward (radius graph + energy + the force
  autograd.grad):
    inductor : create_graph=False as in the eager rows, via the same no_create_graph() hook on
               torch.autograd.grad, installed once for the run and wrapped in
               torch._dynamo.disable so Dynamo graph-breaks cleanly at it; AOTAutograd's
               compiled backward computes the forces.
    fused    : torch._dynamo.allow_in_graph(torch.autograd.grad) with the upstream
               create_graph=True (no hook), so the force backward is inlined into ONE compiled
               graph (MACE's upstream recipe, generalised); the outputs are detached inside the
               compiled callable (lc.fused_callable) so AOTAutograd compiles an inference graph
               run under no_grad, like ATOMTRICK's make_fx step.  create_graph=True is recorded.
  --create-graph is ignored with --compile (the recipe fixes it).  Both: PYTORCH_JIT=0 before
  import torch, e3nn.set_optimization_defaults(jit_script_fx=False) before the model is built,
  parameters frozen (requires_grad_(False)), batch.pos.requires_grad_(True) set by the harness,
  10 warmup (trace + compile, excluded) + 50 timed calls as for eager rows; after the warmup
  the compiled energy + forces are compared with one eager call of the same model
  (compile_max_rel_err; a mismatch is a compile_error).  Extra CSV columns: compile,
  compile_graphs_warmup, compile_graphs_timed, compile_graph_breaks, compile_warmup_s,
  compile_retimed, compile_max_rel_err, pytorch_jit.  A compile failure is recorded as status
  "compile_error: ..." (no eager fallback).
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
from torch_geometric.data import Data, Batch  # noqa: E402
from torch_cluster import radius_graph  # noqa: E402

# The model code lives in the Equiformer clone given by --repo-root; it is imported in
# main() after argument parsing (the original script sat inside that clone).
GraphAttentionTransformerMD17 = None

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

# Processed-data file stems (ATOMTRICK layout, produced by scripts/data/ of the ATOMTRICK repo).
_STEM = {"Ac_Ala3_NHMe": "Ac-Ala3-NHMe", "AT_AT": "AT-AT", "AT_AT_CG_CG": "AT-AT-CG-CG",
         "buckyball_catcher": "buckyball-catcher",
         "double_walled_nanotube": "double-walled_nanotube"}


def load_real_frames(data_root, dataset, molecule, n, split="test"):
    """First ``n`` frames of ``<data_root>/<dataset>/<stem>_<split>_default.npz`` as a list of
    (z LongTensor (na,), pos FloatTensor (na,3)) -- the same frames bench_atomtrick.py uses."""
    import numpy as np
    stem = _STEM.get(molecule, molecule)
    d = np.load(os.path.join(data_root, dataset, f"{stem}_{split}_default.npz"))
    N = np.asarray(d["N"]).reshape(-1)
    na = int(N[0])
    Z = np.asarray(d["Z"], dtype=np.int64).reshape(len(N), na)
    R = np.asarray(d["R"], dtype=np.float32).reshape(len(N), na, 3)
    return [(torch.from_numpy(Z[i % len(N)].copy()), torch.from_numpy(R[i % len(N)].copy()))
            for i in range(n)]


def build_model(device):
    model = GraphAttentionTransformerMD17(
        irreps_in='64x0e',
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=5.0,
        number_of_basis=32, fc_neurons=[64, 64], basis_type='exp',
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
    ).to(device).eval()
    return model


def make_synthetic_batch(n_atoms, batch_size, device):
    data_list = []
    for _ in range(batch_size):
        pos = torch.randn(n_atoms, 3, device='cpu') * 3.0
        z = torch.randint(1, 10, (n_atoms,), device='cpu')
        d = Data(pos=pos, z=z)
        data_list.append(d)
    batch = Batch.from_data_list(data_list).to(device)
    return batch


def make_real_batch(data_root, dataset, molecule, batch_size, device, coords="real",
                    raw_root=None):
    frames = (lc.load_raw_frames(raw_root, dataset, molecule, batch_size) if coords == "raw"
              else load_real_frames(data_root, dataset, molecule, batch_size))
    return Batch.from_data_list([Data(pos=pos, z=z) for z, pos in frames]).to(device)


def timed_forwards(model, batch, warmup, iters, call=None):
    call = call or (lambda: model(batch.z, batch.pos, batch.batch))
    for _ in range(warmup):
        _ = call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _ = call()
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
    ap.add_argument("--output", default="results/equiformer_md22_latency.csv")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4],
                    help="default 1 4 (reported); the original harness defaulted to 4 128")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters",  type=int, default=50)
    ap.add_argument("--molecules", nargs="+", default=None)
    ap.add_argument("--dataset", default="md22", choices=["md22", "rmd17"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo-root", default=os.environ.get("EQUIFORMER_ROOT"),
                    help="Equiformer clone (atomicarchitects/equiformer@64cb786 + "
                         "nets_init.patch); env EQUIFORMER_ROOT")
    ap.add_argument("--coords", choices=["synthetic", "real", "raw"], default="raw",
                    help="raw = reported protocol (frames 0..bs-1 of the raw downloads under "
                         "--raw-root); synthetic = random N(0,3A) coordinates (not used for "
                         "reported rows); real = ATOMTRICK processed test frames from --data-root")
    ap.add_argument("--data-root", default="data",
                    help="ATOMTRICK processed-data root (only for --coords real)")
    ap.add_argument("--create-graph", choices=["true", "false"], default="false",
                    help="false = inference setting, no second-order graph (reported); true = "
                         "upstream create_graph=True force grad; ignored "
                         "with --compile (inductor -> false, fused -> true)")
    lc.add_common_args(ap)
    args = ap.parse_args()
    fused = args.compile == "fused"
    create_graph = fused if args.compile != "none" else (args.create_graph == "true")
    if args.raw_root:
        args.raw_root = os.path.abspath(args.raw_root)
    if args.coords == "raw" and not args.raw_root:
        ap.error("--coords raw (the default) needs --raw-root (or env LATENCY_RAW_ROOT)")
    if not args.repo_root:
        ap.error("--repo-root (or env EQUIFORMER_ROOT) is required")
    args.output = os.path.abspath(args.output)
    args.data_root = os.path.abspath(args.data_root)
    repo_root = str(Path(args.repo_root).resolve())
    sys.path.insert(0, repo_root)
    if args.compile != "none":   # e3nn codegen must stay fx / Python for Dynamo: set before any
        print(f"compile={args.compile}: PYTORCH_JIT={lc.pytorch_jit_state()} "   # e3nn module
              f"e3nn_jit_script_fx_off={lc.e3nn_disable_script()}", flush=True)  # is built
    global GraphAttentionTransformerMD17
    from nets.graph_attention_transformer_md17 import GraphAttentionTransformerMD17

    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB)", flush=True)

    os.chdir(repo_root)   # as in the original harness (it ran from inside the clone)
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
        "gpu_name", "coords", "status",
    ] + [k for k in lc.PROV_FIELDS if k != "coords"]

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
            "coords": args.coords,
            "status": status,
            "latency_mean_ms": None, "latency_median_ms": None,
            "latency_p10_ms": None, "latency_p90_ms": None,
            "latency_std_ms": None, "throughput_samples_per_s": None,
            "peak_mem_gb": None,
            **prov,
        }

    # --compile inductor: the create_graph=False hook is installed ONCE for the whole run (its
    # identity is guarded by Dynamo) and dynamo-disabled so the graph breaks cleanly at it.
    compiled = cstats = hook = hook_ctx = None
    compile_setup_error = None      # a failure here is a compile failure of every row
    if args.compile != "none":
        try:
            print(f"dynamo={lc.configure_dynamo()}", flush=True)
            print(f"  autograd.grad allowed in graph: {lc.set_autograd_in_graph(fused)}", flush=True)
            cstats = lc.CompileStats()
            if not fused:
                hook_ctx = lc.no_create_graph(True, dynamo_disable=True)
                hook = hook_ctx.__enter__()
        except Exception as e:
            print(f"COMPILE SETUP FAILED: {e}", flush=True)
            traceback.print_exc()
            compile_setup_error = lc.short_msg(e)

    try:
        print("\nBuilding Equiformer model (nonlinear_exp_L2, 6 layers)...", flush=True)
        model = build_model(device)
        prov = dict(lc.set_precision(args.matmul_precision), param_dtype=lc.param_dtype(model),
                    forces="autograd", create_graph=create_graph, **lc.compile_prov(args.compile))
        if args.compile != "none" and compile_setup_error is None:
            try:
                lc.freeze_params(model)
                compiled = lc.compile_callable(lc.fused_callable(model) if fused else model)
            except Exception as e:
                print(f"COMPILE SETUP FAILED: {e}", flush=True)
                traceback.print_exc()
                compile_setup_error = lc.short_msg(e)
        print(f"precision: {prov}", flush=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {n_params:,}", flush=True)

        cuda_dead = None   # a sticky CUDA error (illegal memory access, device-side assert) kills the
        #                  # context: the remaining rows are recorded as not run instead of crashing the file
        for mol, n_atoms in mols:
            print(f"\n=== {mol} ({n_atoms} atoms) ===", flush=True)
            if compile_setup_error is not None:
                for bs in args.batch_sizes:
                    emit(blank_row(mol, n_atoms, bs, n_params=n_params,
                                   status=f"compile_error: {compile_setup_error}"))
                continue

            for bs in args.batch_sizes:
                row = blank_row(mol, n_atoms, bs, n_params=n_params, status="ok")
                if cuda_dead is not None:
                    row["status"] = (("compile_error: " if compiled is not None else "error: ")
                                     + f"not run, CUDA context lost earlier ({cuda_dead})")
                    emit(row)
                    continue
                try:
                    batch = (make_synthetic_batch(n_atoms, bs, device)
                             if args.coords == "synthetic" else
                             make_real_batch(args.data_root, args.dataset, mol, bs, device,
                                             args.coords, args.raw_root))
                    row["edges"] = int(radius_graph(batch.pos, r=5.0, batch=batch.batch,
                                                    max_num_neighbors=1000).shape[1])
                    row["graph_build_median_ms"] = lc.median_cuda_ms(
                        lambda: radius_graph(batch.pos, r=5.0, batch=batch.batch,
                                             max_num_neighbors=1000))
                    if compiled is None:
                        with lc.no_create_graph(args.create_graph == "false") as ncg:
                            times_ms, peak_bytes = timed_forwards(
                                model, batch, args.warmup, args.iters,
                            )
                        if args.create_graph == "false" and ncg["rewritten"] == 0:
                            raise RuntimeError("--create-graph false had no effect "
                                               f"({ncg['calls']} autograd.grad calls seen)")
                    else:
                        batch.pos.requires_grad_(True)   # forward's requires_grad_ then is a no-op
                        seen = hook["rewritten"] if hook is not None else 0
                        times_ms, peak_bytes, cinfo = lc.compiled_timed_forwards(
                            lambda: compiled(batch.z, batch.pos, batch.batch),
                            args.warmup, args.iters, cstats,
                            # output check: the same (un-compiled) model on the same batch; the
                            # forward returns (energy, forces)
                            eager_call=lambda: model(batch.z, batch.pos, batch.batch),
                            pick=tuple,
                        )
                        if hook is not None and hook["rewritten"] == seen:
                            raise RuntimeError("--compile inductor: the create_graph=False hook "
                                               f"was not reached ({hook['calls']} autograd.grad "
                                               "calls seen)")
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

                batch = None
                if cuda_dead is None:   # every CUDA call fails on a lost context
                    torch.cuda.empty_cache()

    finally:
        if hook_ctx is not None:
            hook_ctx.__exit__(None, None, None)
        csv_file.close()
        print(f"\nCSV saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
