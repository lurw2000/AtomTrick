#!/usr/bin/env python3
"""Measure MACE inference latency on MD22 / rMD17.

Mirrors scripts/benchmark_inference.py from the GotenNet repo so the output
CSVs can be compared side-by-side. Each (molecule, batch_size) configuration
times energy + forces-via-autograd forward passes on a fresh random-init
ScaleShiftMACE model.

MACE paper config (Kovács et al. 2023, J. Chem. Phys. 159, 044118, "b. MD22 models"):
  ScaleShiftMACE 256-2 @ 2×5 Å
    r_max=5.0, num_interactions=2, max_ell=3, num_bessel=8, num_polynomial_cutoff=5
    hidden_irreps="256x0e + 256x1o + 256x2e" (L=2, 256 channels)
    MLP_irreps="16x0e", correlation=3, radial_MLP=[64,64,64]
    interaction: RealAgnosticResidualInteractionBlock (RealAgnosticInteractionBlock first)
  → ~3.77M params
  Paper trains at float64, but we run float32 to match the GotenNet benchmark.

MACE computes forces as -dE/dx via autograd when compute_force=True (default).
So we must NOT wrap in no_grad(); model.eval() is the right switch.

Output: results/mace_<dataset>_latency_<gpu>.csv (same schema as gotennet_md22_latency_h100.csv).

--- ATOMTRICK release notes (benchmarks/latency/baselines/mace/) -------------------------
Harness for the MACE rows of the latency tables.  --data-root is a required argument (or env
MACE_BENCH_DATA_ROOT).  The DEFAULTS reproduce the reported numbers (float32, autograd forces
with create_graph=False, real frames).
  * MACE = ACEsuit/mace at 667eee4 (v0.3.15 + 46 commits; unmodified library code -- the local
    commit only added this script).  Random-init ScaleShiftMACE, 3,769,616 params (MD22 z-table).
  * Real frames 0..bs-1 of the raw trajectories; the neighbour list is built in build_batch,
    OUTSIDE the timed region (like ATOMTRICK).  Eager PyTorch, strict fp32 (no TF32),
    cuEquivariance off.  The model is NOT compiled.

--- Protocol switches (defaults = the reported protocol) -------------------------------------
  * --dtype {float32,float64} (default float32 = reported; MACE itself trains in float64).
    Sets torch's default dtype before the model and the batch are built.
  * --matmul-precision {highest,high,medium} (default highest = strict fp32), set before timing.
  * Forces: model(..., training=False, compute_force=True) = autograd -dE/dx with
    create_graph=False (unchanged).  Extra CSV columns: edges, provenance.

--- --compile {none,inductor,fused} (default none = the eager rows above, unchanged) ----------
  Times the SAME model under torch.compile(dynamic=True) (TorchInductor, default mode, no CUDA
  graphs).  The model is built the way MACE's own prepare() (mace/tools/compile.py) builds it
  -- e3nn TorchScript codegen off (disable_e3nn_codegen) and the registered modules
  fx-simplified (simplify) -- but prepare() itself is not called: its allow_autograd=False
  branch calls torch._dynamo.disallow_in_graph(torch.autograd.grad), which raises on torch 2.5
  because autograd.grad was never allowed (its default rule is already a graph break).  The
  compiled object is the whole model call (energy forward + force autograd.grad):
    inductor : called with training=False (create_graph=False, as eager); Dynamo graph-breaks
               at autograd.grad and AOTAutograd's compiled backward computes the forces.
    fused    : MACE's upstream torch.compile recipe (the calculator's): allow_in_graph(
               autograd.grad), called with training=True (create_graph=True), so the force
               backward is inlined into ONE compiled graph; the outputs are detached inside the
               compiled callable (lc.fused_callable) so AOTAutograd compiles an inference graph
               run under no_grad, like ATOMTRICK's make_fx step.  create_graph=True is recorded.
  Both: PYTORCH_JIT=0 before import torch, parameters frozen (requires_grad_(False)),
  positions.requires_grad_(True) set by the harness (MACE's prepare_graph skips it under
  torch.compile), 10 warmup (trace + compile, excluded) + 50 timed calls as for eager rows.
  After the warmup the compiled energy + forces are compared with one eager call of the same
  model (compile_max_rel_err; MACE's compute_forces(allow_unused=True) would otherwise emit
  ZERO forces if a graph break ever separated positions from the energy).  The compiled model is
  built ONCE per run (latency is weight-agnostic) so the dynamic-shape graph is reused across
  molecules; eager rows keep one fresh model per molecule.  Extra CSV columns: compile,
  compile_graphs_warmup, compile_graphs_timed, compile_graph_breaks, compile_warmup_s,
  compile_retimed, compile_max_rel_err, pytorch_jit.  A compile failure is recorded as status
  "compile_error: ..." (no eager fallback).  torch 2.5.1 has no torch.compile on Python 3.13,
  so the compiled rows need a Python <= 3.12 copy of the MACE environment.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from e3nn import o3  # noqa: E402

import ase  # noqa: E402
from mace import data as mace_data  # noqa: E402
from mace import modules, tools  # noqa: E402
from mace.tools import torch_geometric  # noqa: E402


MD22_MOLECULES = [
    ("Ac_Ala3_NHMe",          42),
    ("DHA",                   56),
    ("AT_AT",                 60),
    ("stachyose",             87),
    ("AT_AT_CG_CG",          118),
    ("buckyball_catcher",    148),
    ("double_walled_nanotube", 370),
]

# MD22 raw .npz file names under <data_root>/raw/md22/
MD22_FILES = {
    "Ac_Ala3_NHMe":           "md22_Ac-Ala3-NHMe.npz",
    "AT_AT":                  "md22_AT-AT.npz",
    "AT_AT_CG_CG":            "md22_AT-AT-CG-CG.npz",
    "DHA":                    "md22_DHA.npz",
    "buckyball_catcher":      "md22_buckyball-catcher.npz",
    "double_walled_nanotube": "md22_double-walled_nanotube.npz",
    "stachyose":              "md22_stachyose.npz",
}

RMD17_MOLECULES = [
    ("ethanol", 9), ("malonaldehyde", 9), ("benzene", 12), ("uracil", 12),
    ("toluene", 15), ("salicylic", 16), ("naphthalene", 18), ("paracetamol", 20),
    ("aspirin", 21), ("azobenzene", 24),
]

DATASETS = {"md22": MD22_MOLECULES, "rmd17": RMD17_MOLECULES}


def load_frames(dataset: str, data_root: str, molecule: str, n_samples: int):
    """Load n_samples frames from the MD22 / rMD17 .npz for the given molecule.

    Returns (ase.Atoms list, atomic_numbers array). rMD17 npz stores atomic
    numbers under 'nuclear_charges' and coordinates (Angstrom) under 'coords';
    MD22 uses 'z' and 'R'.
    """
    if dataset == "md22":
        path = os.path.join(data_root, "raw", "md22", MD22_FILES[molecule])
        d = np.load(path)
        z = np.asarray(d["z"])                # (n_atoms,)
        R = np.asarray(d["R"])[:n_samples]    # (N, n_atoms, 3)
    else:  # rmd17
        path = os.path.join(data_root, "raw", "rmd17", "npz_data",
                            f"rmd17_{molecule}.npz")
        d = np.load(path)
        z = np.asarray(d["nuclear_charges"])       # (n_atoms,)
        R = np.asarray(d["coords"])[:n_samples]    # (N, n_atoms, 3)
    atoms_list = []
    for pos in R:
        atoms_list.append(ase.Atoms(numbers=z, positions=pos))
    return atoms_list, z


def build_model(z_table: tools.AtomicNumberTable, r_max: float = 5.0) -> torch.nn.Module:
    """Build MACE ScaleShiftMACE 256-2 (paper config, fresh random weights)."""
    return modules.ScaleShiftMACE(
        r_max=r_max,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=len(z_table.zs),
        hidden_irreps=o3.Irreps("256x0e + 256x1o + 256x2e"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(len(z_table.zs)),   # placeholder; latency is weight-agnostic
        avg_num_neighbors=20.0,                       # placeholder
        atomic_numbers=z_table.zs,
        correlation=3,
        radial_type="bessel",
        radial_MLP=[64, 64, 64],
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


def build_batch(atoms_list, z_table, r_max, device):
    """Wrap ase.Atoms list → AtomicData → batched dict on GPU."""
    configs = [mace_data.config_from_atoms(a) for a in atoms_list]
    dataset = [mace_data.AtomicData.from_config(c, z_table=z_table, cutoff=r_max)
               for c in configs]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False, drop_last=False,
    )
    batch = next(iter(loader))
    batch = batch.to(device)
    return batch.to_dict()


def timed_forwards(model, batch_dict, warmup: int, iters: int):
    """Warm up, then time forward passes with torch.cuda.Event.

    MACE forward(compute_force=True) triggers autograd internally for the force
    gradient — don't wrap in no_grad().
    """
    for _ in range(warmup):
        _ = model(batch_dict, training=False, compute_force=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _ = model(batch_dict, training=False, compute_force=True)
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
    ap.add_argument("--data-root", default=os.environ.get("MACE_BENCH_DATA_ROOT"),
                    help="raw .npz root (expects raw/md22/*.npz and raw/rmd17/npz_data/*.npz); "
                         "env MACE_BENCH_DATA_ROOT")
    ap.add_argument("--output", default="results/mace_md22_latency.csv")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--molecules", nargs="+", default=None,
                    help="Subset of molecules; default = all in the dataset")
    ap.add_argument("--dataset", default="md22", choices=["md22", "rmd17"])
    ap.add_argument("--r-max", type=float, default=5.0,
                    help="MACE cutoff (paper 2x5 Å variant uses 5.0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="model + input dtype (reported latency: float32)")
    lc.add_common_args(ap)
    args = ap.parse_args()
    torch.set_default_dtype(getattr(torch, args.dtype))
    if not args.data_root:
        ap.error("--data-root (or env MACE_BENCH_DATA_ROOT) is required")

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

    # Build a single z_table from the union of z's across all target molecules
    zs_union = set()
    for m, _ in mols:
        atoms_one, z_arr = load_frames(args.dataset, args.data_root, m, 1)
        zs_union.update(int(z) for z in z_arr)
    z_table = tools.AtomicNumberTable(sorted(zs_union))
    print(f"Atomic number table: {z_table.zs}", flush=True)

    compiled = cstats = None
    fused = args.compile == "fused"

    fieldnames = [
        "molecule", "atoms", "batch_size", "total_atoms_per_batch", "n_params",
        "warmup", "iters",
        "latency_mean_ms", "latency_median_ms", "latency_p10_ms",
        "latency_p90_ms", "latency_std_ms",
        "throughput_samples_per_s", "peak_mem_gb",
        "gpu_name", "status",
    ] + lc.PROV_FIELDS

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

    # --compile: one model for the whole run, prepared and compiled MACE's way.  A failure here
    # is a compile failure of every row (no eager fallback).
    compile_setup_error = None
    if args.compile != "none":
        try:
            # the two build steps of MACE's prepare(): e3nn codegen off while building, then
            # fx-simplify the registered modules (prepare() itself is not called, see docstring)
            from mace.tools.compile import disable_e3nn_codegen, simplify
            print(f"compile={args.compile}: PYTORCH_JIT={lc.pytorch_jit_state()} "
                  f"dynamo={lc.configure_dynamo()}", flush=True)
            with disable_e3nn_codegen():
                model = build_model(z_table, r_max=args.r_max)
            model = simplify(model)
            model = lc.freeze_params(model.to(device).eval())
            # full provenance before the compile steps, so compile_error rows carry it too
            prov = dict(lc.set_precision(args.matmul_precision), param_dtype=lc.param_dtype(model),
                        forces="autograd", create_graph=fused, coords="raw",
                        **lc.compile_prov(args.compile))
            print(f"  autograd.grad allowed in graph: {lc.set_autograd_in_graph(fused)}", flush=True)
            compiled = lc.compile_callable(lc.fused_callable(model) if fused else model)
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
            try:
                if args.compile == "none":
                    model = build_model(z_table, r_max=args.r_max).to(device).eval()
                n_params = sum(p.numel() for p in model.parameters())
                prov = dict(lc.set_precision(args.matmul_precision),
                            param_dtype=lc.param_dtype(model), forces="autograd",
                            create_graph=fused, coords="raw", **lc.compile_prov(args.compile))
                print(f"  model params: {n_params:,}  precision: {prov}", flush=True)
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
                    atoms_list, _ = load_frames(args.dataset, args.data_root, mol, bs)
                    # If the dataset has fewer than bs samples, pad by repetition
                    # so the batch matches the requested size.
                    while len(atoms_list) < bs:
                        atoms_list.append(atoms_list[len(atoms_list) % max(1, len(atoms_list))])
                    batch_dict = build_batch(atoms_list, z_table, args.r_max, device)
                    row["edges"] = int(batch_dict["edge_index"].shape[1])
                    if batch_dict["positions"].dtype != getattr(torch, args.dtype):
                        raise RuntimeError(f"positions dtype {batch_dict['positions'].dtype}")

                    if compiled is None:
                        times_ms, peak_bytes = timed_forwards(
                            model, batch_dict, args.warmup, args.iters,
                        )
                    else:
                        # prepare_graph() does not set this while compiling
                        batch_dict["positions"].requires_grad_(True)
                        times_ms, peak_bytes, cinfo = lc.compiled_timed_forwards(
                            lambda: compiled(batch_dict, training=fused, compute_force=True),
                            args.warmup, args.iters, cstats,
                            # output check: the same (un-compiled) model, same batch and recipe
                            eager_call=lambda: model(batch_dict, training=fused, compute_force=True),
                            pick=lambda o: (o["energy"], o["forces"]),
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
