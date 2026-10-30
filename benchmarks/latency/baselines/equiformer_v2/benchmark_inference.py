#!/usr/bin/env python3
"""Measure EquiformerV2 inference latency on MD22.

For each MD22 molecule and batch size, instantiates the model (fresh random
weights -- latency depends on architecture + input shape, not trained params),
creates a batch (raw-trajectory frames by default; see --coords), runs warmup
iterations, then times many forward passes end-to-end with torch.cuda.Event.

Upstream EquiformerV2 computes forces via a dedicated force prediction block
(not autograd; --forces direct), but the forward() method is decorated with
@conditional_grad (torch.enable_grad()), so we must NOT wrap timing in
torch.no_grad(). model.eval() is used to disable dropout.

Output: results/equiformer_v2_md22_latency.csv

--- ATOMTRICK release notes (benchmarks/latency/baselines/equiformer_v2/) ----------------
Harness for the EquiformerV2 rows of the latency tables.  Paths are CLI arguments, and the
switches below select the inputs and the force protocol.  The DEFAULTS reproduce the reported
numbers.
  * Model = the OC20 "N@12_L@6_M@2" configuration, otf_graph=True, max_neighbors=50.  It was
    never trained on rMD17/MD22; it is timed as an architecture-cost reference only.
  * --repo-root : a clone of atomicarchitects/equiformer_v2 @ d5ad4be (unmodified) with the
    minimal OCP stub in ./ocpmodels_shim importable (PYTHONPATH=<this dir>/ocpmodels_shim).
  * --forces {autograd,direct} (default autograd = reported).  autograd builds the model with
    regress_forces=False (no force head; 77,182,465 parameters) and computes F = -dE/dpos with
    autograd.grad, create_graph=False -- the same force definition as every other model.
    direct = the DIRECT force head (83,164,802 parameters; not used for any reported row).
  * --coords raw --raw-root R (DEFAULT, = reported numbers): frames 0..bs-1 of the raw rMD17 /
    MD22 downloads, the same inputs as the MACE / GotenNet harnesses and
    bench_atomtrick.py --frame_source raw.
    --coords synthetic: pos ~ N(0, 3 A), z ~ U{1..9} (not like-for-like; not used for any
    reported row).
    --coords real --data-root data : ATOMTRICK processed test frames (same as bench_atomtrick.py).
  * --matmul-precision {highest,high,medium} (default highest = strict fp32), set before timing.
  * Extra CSV columns: edges (radius graph the model builds, same r / neighbour cap),
    graph_build_median_ms (that radius_graph call timed alone), precision / dtype / force /
    version provenance.
  * The radius graph is built INSIDE the model (timed).  Eager PyTorch, strict fp32 (no TF32).

--- --compile {none,inductor,fused} (default none = the eager rows above, unchanged) ----------
  Times the SAME model under torch.compile(dynamic=True) (TorchInductor, default mode, no CUDA
  graphs).  The compiled object is the whole timed callable: for --forces autograd the
  harness's energy + autograd.grad function (model forward incl. the on-the-fly radius graph,
  then F = -dE/dpos), for --forces direct the model itself:
    inductor : create_graph=False as in the eager rows; Dynamo graph-breaks at autograd.grad
               and AOTAutograd's compiled backward computes the forces.
    fused    : torch._dynamo.allow_in_graph(torch.autograd.grad) with create_graph=True in the
               harness call, so the force backward is inlined into ONE compiled graph (MACE's
               upstream recipe, generalised).  create_graph=True is recorded.
  Both: PYTORCH_JIT=0 before import torch, parameters frozen (requires_grad_(False)),
  batch.pos.requires_grad_(True), 10 warmup (trace + compile, excluded) + 50 timed calls as for
  eager rows; after the warmup the compiled energy + forces are compared with one eager call of
  the same model (compile_max_rel_err; a mismatch is a compile_error).  fused detaches the
  outputs inside the compiled callable so AOTAutograd compiles an inference graph run under
  no_grad, like ATOMTRICK's make_fx step.  The radius graph (torch_cluster, a custom op
  without a fake kernel) is built inside the model's forward, so the graph break it causes
  lands at the model call and the inlined autograd.grad of the fused recipe may not reach
  batch.pos -> such rows are compile_error.  Extra CSV columns: compile,
  compile_graphs_warmup, compile_graphs_timed, compile_graph_breaks, compile_warmup_s,
  compile_retimed, compile_max_rel_err, pytorch_jit.  A compile failure is recorded as status
  "compile_error: ..." (no eager fallback).  Model-specific: the two cache helpers of the
  shared CoefficientMappingModule run outside the compiled graph (opaque_cache_helpers;
  Dynamo mis-traces their single-entry attribute cache and the compiled model is then wrong).

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --molecules AT_AT DHA
    python scripts/benchmark_inference.py --batch-sizes 1 4 16
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

# Imported in main() from the clone given by --repo-root (the original script sat inside it).
EquiformerV2_OC20 = None

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


def build_model(device, regress_forces=True):
    model = EquiformerV2_OC20(
        num_atoms=None,
        bond_feat_dim=None,
        num_targets=None,
        use_pbc=False,
        regress_forces=regress_forces,
        otf_graph=True,
        max_neighbors=50,
        max_radius=5.0,
        max_num_elements=90,
        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        norm_type='layer_norm_sh',
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=18,
        num_sphere_samples=128,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function='gaussian',
        num_distance_basis=512,
        attn_activation='silu',
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation='silu',
        use_gate_act=False,
        use_grid_mlp=True,
        use_sep_s2_act=True,
        alpha_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        weight_init='uniform',
    ).to(device).eval()
    return model


def make_synthetic_batch(n_atoms, batch_size, device):
    data_list = []
    for _ in range(batch_size):
        pos = torch.randn(n_atoms, 3, device='cpu') * 3.0
        z = torch.randint(1, 10, (n_atoms,), device='cpu')
        natoms = torch.tensor([n_atoms])
        d = Data(pos=pos, atomic_numbers=z, natoms=natoms)
        data_list.append(d)
    batch = Batch.from_data_list(data_list).to(device)
    return batch


def make_real_batch(data_root, dataset, molecule, batch_size, device, coords="real",
                    raw_root=None):
    frames = (lc.load_raw_frames(raw_root, dataset, molecule, batch_size) if coords == "raw"
              else load_real_frames(data_root, dataset, molecule, batch_size))
    return Batch.from_data_list([
        Data(pos=pos, atomic_numbers=z, natoms=torch.tensor([len(z)])) for z, pos in frames
    ]).to(device)


def make_call(model, batch, forces):
    """One timed call = energy + forces.  direct: the model's force head;
    autograd: F = -dE/dpos, create_graph=False (model built with regress_forces=False)."""
    if forces == "direct":
        return lambda: model(batch)
    batch.pos.requires_grad_(True)

    def call():
        energy = model(batch)
        f = -torch.autograd.grad(energy.sum(), batch.pos, create_graph=False)[0]
        return energy, f
    return call


def opaque_cache_helpers(model):
    """--compile: keep CoefficientMappingModule's cache helpers (coefficient_idx,
    get_rotate_inv_rescale) OUT of the compiled graph (torch._dynamo.disable on the bound
    methods, i.e. a graph break around each call; they run eagerly, as in the eager rows).
    They keep a single-entry cache in plain attributes (lmax_cache / mmax_cache /
    mask_indices_cache / rotate_inv_rescale_cache, overwritten on every key change) that is
    shared by all 12 blocks; traced by Dynamo, the compiled model returned wrong values
    (energy 1-2 %, forces up to 76 % off eager with torch 2.5.1 -- every Inductor knob and
    AOT-eager were checked: AOT-eager matched, so it is the trace of these helpers), while
    with them outside the graph the compiled model matches eager to < 1e-6 (compile_max_rel_err).
    Returns the number of methods wrapped."""
    import torch._dynamo
    n = 0
    for mod in model.modules():
        for meth in ("coefficient_idx", "get_rotate_inv_rescale"):
            fn = getattr(mod, meth, None)
            if fn is not None and not getattr(fn, "_latency_opaque", False):
                w = torch._dynamo.disable(fn)
                try:
                    w._latency_opaque = True
                except AttributeError:
                    pass
                setattr(mod, meth, w)
                n += 1
    return n


def make_compiled(model, forces, create_graph):
    """--compile: the object torch.compile wraps.  autograd: one function (energy forward +
    autograd.grad, create_graph per recipe) taking the batch, so the dynamic-shape graph is
    shared by every (molecule, batch size); direct: the model itself."""
    if forces == "direct":
        return lc.compile_callable(lc.fused_callable(model) if create_graph else model)

    def energy_and_forces(batch):
        energy = model(batch)
        f = -torch.autograd.grad(energy.sum(), batch.pos, create_graph=create_graph)[0]
        if create_graph:       # fused: detached outputs -> AOTAutograd inference graph
            return energy.detach(), f.detach()
        return energy, f
    return lc.compile_callable(energy_and_forces)


def timed_forwards(model, batch, warmup, iters, call=None):
    call = call or (lambda: model(batch))
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
    ap.add_argument("--output", default="results/equiformer_v2_md22_latency.csv")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4],
                    help="default 1 4 (reported); the original harness defaulted to 4 128")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters",  type=int, default=50)
    ap.add_argument("--molecules", nargs="+", default=None)
    ap.add_argument("--dataset", default="md22", choices=["md22", "rmd17"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo-root", default=os.environ.get("EQUIFORMER_V2_ROOT"),
                    help="EquiformerV2 clone (atomicarchitects/equiformer_v2@d5ad4be); "
                         "env EQUIFORMER_V2_ROOT")
    ap.add_argument("--coords", choices=["synthetic", "real", "raw"], default="raw",
                    help="raw = reported protocol (frames 0..bs-1 of the raw downloads under "
                         "--raw-root); synthetic = random N(0,3A) coordinates (not used for "
                         "reported rows); real = ATOMTRICK processed test frames from --data-root")
    ap.add_argument("--data-root", default="data",
                    help="ATOMTRICK processed-data root (only for --coords real)")
    ap.add_argument("--forces", choices=["direct", "autograd"], default="autograd",
                    help="autograd = -dE/dpos, create_graph=False, no force head (reported); "
                         "direct = the model's force head (not used for reported rows)")
    lc.add_common_args(ap)
    args = ap.parse_args()
    fused = args.compile == "fused"
    create_graph = fused and args.forces == "autograd"
    if args.raw_root:
        args.raw_root = os.path.abspath(args.raw_root)
    if args.coords == "raw" and not args.raw_root:
        ap.error("--coords raw (the default) needs --raw-root (or env LATENCY_RAW_ROOT)")
    if not args.repo_root:
        ap.error("--repo-root (or env EQUIFORMER_V2_ROOT) is required")
    sys.path.insert(0, str(Path(args.repo_root).resolve()))
    global EquiformerV2_OC20
    from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB)", flush=True)

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
        print("\nBuilding EquiformerV2 model (N@12_L@6_M@2)...", flush=True)
        model = build_model(device, regress_forces=(args.forces == "direct"))
        prov = dict(lc.set_precision(args.matmul_precision), param_dtype=lc.param_dtype(model),
                    forces="direct-head" if args.forces == "direct" else "autograd",
                    create_graph=create_graph, **lc.compile_prov(args.compile))
        if args.compile != "none" and compile_setup_error is None:
            try:
                print(f"  cache helpers kept outside the graph: {opaque_cache_helpers(model)}", flush=True)
                compiled = make_compiled(lc.freeze_params(model), args.forces, create_graph)
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
                                                    max_num_neighbors=50).shape[1])
                    row["graph_build_median_ms"] = lc.median_cuda_ms(
                        lambda: radius_graph(batch.pos, r=5.0, batch=batch.batch,
                                             max_num_neighbors=50))
                    if compiled is None:
                        times_ms, peak_bytes = timed_forwards(
                            model, batch, args.warmup, args.iters,
                            call=make_call(model, batch, args.forces),
                        )
                    else:
                        if args.forces != "direct":
                            batch.pos.requires_grad_(True)
                        times_ms, peak_bytes, cinfo = lc.compiled_timed_forwards(
                            lambda: compiled(batch), args.warmup, args.iters, cstats,
                            # output check: the eager call of the same model on the same batch
                            # ((energy, forces) for both force modes)
                            eager_call=make_call(model, batch, args.forces),
                            pick=lambda o: tuple(o) if isinstance(o, (tuple, list)) else (o,),
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

                batch = None
                if cuda_dead is None:   # every CUDA call fails on a lost context
                    torch.cuda.empty_cache()

    finally:
        csv_file.close()
        print(f"\nCSV saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
