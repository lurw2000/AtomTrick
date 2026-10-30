#!/usr/bin/env python3
"""GPU inference-latency benchmark for ATOMTRICK: eager autograd vs make_fx + torch.compile.

What is timed
-------------
One call = energy + forces (F = -dE/dR) for a batch of ``bs`` structures of one molecule.

* ``eager``  : ``model(batch)`` -- the plain PyTorch autograd force path (grad ON; the model
               enables grad internally, forces need it).
* ``makefx`` : :class:`AtomTrick.backward.makefx_compiled.MakeFXCompiledStep` called under
               ``torch.no_grad()``.  ``make_fx`` traces forward + force backward into one
               aten graph (symbolic shapes), which ``torch.compile(dynamic=True)`` / Inductor
               then compiles.  This is the compiled path of the paper: the fused kernels are
               **Inductor-generated**, there are no hand-written CUDA kernels and no CUDA
               graphs.  One graph is traced per batch size on the first
               molecule and reused for every molecule (atom/edge counts are dynamic).

Protocol (that of the reported numbers): ``--warmup`` untimed calls (they absorb the
make_fx trace + Inductor compile), then ``--iters`` calls, each bracketed by a pair of
``torch.cuda.Event`` s, one synchronize at the end.  The *same* batch (``bs`` consecutive
frames of the chosen split, neighbour list built on the CPU with a 5 A cKDTree **outside**
the timed region) is reused for every call.  Reported per row:

* ``latency_median_ms``                -- median of the per-call times (the headline number)
* ``latency_mean_ms`` / p10 / p90 / std
* ``throughput_mean_structs_per_s``    -- ``bs * 1000 / mean``   (spike-sensitive)
* ``throughput_median_structs_per_s``  -- ``bs * 1000 / median`` (the paper's definition;
                                          robust to spikes)
* ``peak_mem_gb``                      -- ``torch.cuda.max_memory_allocated()`` over the timed
                                          calls (reset after warmup; includes the weights)

Model
-----
Built from the same YAML configs ``train.py`` consumes, resolved by ``train.parse_args``
(``--config``, later files override earlier ones, argparse defaults = the paper model;
default ``configs/rmd17/atomtrick.yaml``, 3,730,368 parameters), through
``Trainer.from_config``.  With ``--checkpoint`` the trained weights are loaded and, unless
``--config`` is given, the architecture is read from the ``args.json`` next to it.  Without
a checkpoint the weights are random (``--seed``), as for the reported latency table --
latency does not depend on the weight values.

Precision
---------
``--tf32 off`` (default) keeps full fp32 matmuls (``torch.set_float32_matmul_precision(
"highest")``), the release default for training and evaluation and the precision the
baseline harnesses in ``baselines/`` run with (PyTorch default), so the comparison is
like-for-like.  ``--tf32 on`` sets ``"high"`` (TF32 matmuls); no reported number uses it.

Input structures
----------------
``--frame_source processed`` (default) reads ``<data_root>/<dataset>/<mol>_<split>_default.npz``
(produced by ``scripts/data/``).  ``--frame_source raw`` reads the raw rMD17
(``<raw_root>/raw/rmd17/npz_data/rmd17_<mol>.npz``) / MD22
(``<raw_root>/raw/md22/md22_<mol>.npz``) downloads -- the same layout the MACE and GotenNet
harnesses in ``baselines/`` read.  The reported set (``results/h100_fp32_2026-10-20/``) used raw
frames 0..bs-1, so this option reproduces its exact edge counts.

Environment set before importing AtomTrick: ``PYTORCH_JIT=0`` (TorchScript breaks the
make_fx second-order trace) and ``ATOMTRICK_MAKEFX_DUCKOFF=1`` (fixed-size molecules can
duck-collide with model constants during symbolic tracing; see makefx_compiled.py).

Examples
--------
    # batch-1 + batch-4 table, all 17 systems, random init, full fp32 (same as the baselines)
    python benchmarks/latency/bench_atomtrick.py --batch_sizes 1 4 \
        --output benchmarks/latency/results/atomtrick_latency.csv
    # TF32 matmuls (not used for any reported number)
    python benchmarks/latency/bench_atomtrick.py --batch_sizes 1 4 --tf32 on
    # trained MD22 weights
    python benchmarks/latency/bench_atomtrick.py \
        --checkpoint <run dir>/<name>_best.pt      # args.json next to it gives the architecture
"""
import os

os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("ATOMTRICK_MAKEFX_DUCKOFF", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from AtomTrick.backward.makefx_compiled import MakeFXCompiledStep  # noqa: E402
from AtomTrick.trainer import Trainer, inference_cutoff  # noqa: E402

DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "configs", "rmd17", "atomtrick.yaml")

# CSV molecule labels (kept identical to the baseline harnesses so merge.py can join rows)
# -> processed / raw file stem.
RMD17 = ["ethanol", "malonaldehyde", "benzene", "uracil", "toluene",
         "salicylic", "naphthalene", "paracetamol", "aspirin", "azobenzene"]
MD22 = ["Ac_Ala3_NHMe", "DHA", "AT_AT", "stachyose", "AT_AT_CG_CG",
        "buckyball_catcher", "double_walled_nanotube"]
MD22_STEM = {
    "Ac_Ala3_NHMe": "Ac-Ala3-NHMe", "DHA": "DHA", "AT_AT": "AT-AT",
    "stachyose": "stachyose", "AT_AT_CG_CG": "AT-AT-CG-CG",
    "buckyball_catcher": "buckyball-catcher",
    "double_walled_nanotube": "double-walled_nanotube",
}
_ALIASES = {v: k for k, v in MD22_STEM.items()}      # accept "Ac-Ala3-NHMe" on the CLI


# --------------------------------------------------------------------------- #
#  Config -> model  (same key names as train.py; later files override earlier)
# --------------------------------------------------------------------------- #
def load_config_files(paths):
    """Resolve model config files the way ``train.py`` does.

    * YAML experiment configs (``configs/...``) go through ``train.parse_args``: argparse
      defaults = the paper model, later files override earlier ones, unknown keys are errors.
    * A run's ``args.json`` is used verbatim; ``Trainer.build_model``'s fallbacks for keys an
      old ``args.json`` lacks then rebuild exactly the model that run trained.
    """
    if not all(p.endswith(".json") for p in paths):
        try:
            import train as _train                      # repo root is on sys.path
            argv = [a for p in paths for a in ("--config", p)] + ["--dry_run"]
            _, ns = _train.parse_args(argv)
            return {k: v for k, v in vars(ns).items() if not k.startswith("_")}
        except (ImportError, AttributeError) as e:      # train.py without parse_args()
            print(f"WARNING: train.parse_args unavailable ({e}); merging configs verbatim",
                  file=sys.stderr)
    cfg = {}
    for p in paths:
        with open(p) as f:
            if p.endswith(".json"):
                d = json.load(f)
            else:
                import yaml
                d = yaml.safe_load(f) or {}
        if not isinstance(d, dict):
            raise ValueError(f"{p}: expected a mapping at the top level")
        cfg.update({k: v for k, v in d.items() if not k.startswith("_")})
    # kv_mode shorthand, resolved the way train.py does
    base = cfg.get("kv_mode") or "o2"
    cfg["k_mode"] = cfg.get("k_mode") or base
    cfg["v_mode"] = cfg.get("v_mode") or base
    return cfg


def build_model(config_paths, checkpoint, device, seed):
    """Build the AtomTrick model through ``Trainer.from_config`` (the same code path
    train.py / evaluate.py use) and optionally load trained weights.

    Returns ``(model, cfg, weights_label)``.
    """
    cfg = load_config_files(config_paths)
    tmp = tempfile.mkdtemp(prefix="atomtrick_bench_")
    cfg.update({
        "device": str(device), "seed": seed, "wandb_project": None,
        "_skip_save_args": True, "out_dir": tmp,   # nothing is written to the repo
    })
    try:
        trainer = Trainer.from_config(cfg)
        label = "random-init"
        if checkpoint:
            raw = torch.load(checkpoint, map_location=device, weights_only=False)  # trusted file
            if isinstance(raw, dict) and "model" in raw:
                # full training checkpoint (*_final.pt / rolling): use the EMA weights
                trainer.load_checkpoint(checkpoint)
                trainer._ema_apply()
            else:
                # *_best.pt: a raw state_dict saved while the EMA weights were applied
                trainer.model.load_state_dict(raw)
            label = os.path.basename(checkpoint)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return trainer.model.eval(), cfg, label


# --------------------------------------------------------------------------- #
#  Input frames + neighbour list (built OUTSIDE the timed region)
# --------------------------------------------------------------------------- #
def _processed_path(data_root, dataset, molecule, split):
    stem = MD22_STEM.get(molecule, molecule)
    return os.path.join(data_root, dataset, f"{stem}_{split}_default.npz")


def load_frames(args, dataset, molecule, n, idx0=0):
    """Return ``n`` consecutive conformations as a list of (z int64 (na,), R float32 (na,3))."""
    if args.frame_source == "processed":
        d = np.load(_processed_path(args.data_root, dataset, molecule, args.split))
        N = np.asarray(d["N"], dtype=np.int64).reshape(-1)
        na = int(N[0])
        if not (N == na).all():
            raise ValueError("processed file mixes atom counts; expected one molecule")
        z_all = np.asarray(d["Z"], dtype=np.int64).reshape(len(N), na)
        R_all = np.asarray(d["R"]).reshape(len(N), na, 3)
        return [(z_all[(idx0 + i) % len(N)],
                 np.asarray(R_all[(idx0 + i) % len(N)], dtype=np.float32))
                for i in range(n)]
    # raw downloads (what the reported set used)
    if dataset == "md22":
        d = np.load(os.path.join(args.raw_root, "raw", "md22",
                                 f"md22_{MD22_STEM[molecule]}.npz"))
        z, R = np.asarray(d["z"], dtype=np.int64), d["R"]
    else:
        d = np.load(os.path.join(args.raw_root, "raw", "rmd17", "npz_data",
                                 f"rmd17_{molecule}.npz"))
        z, R = np.asarray(d["nuclear_charges"], dtype=np.int64), d["coords"]
    return [(z, np.asarray(R[(idx0 + i) % len(R)], dtype=np.float32)) for i in range(n)]


def compute_edges(R_np, cutoff):
    pairs = cKDTree(R_np).query_pairs(cutoff, output_type="ndarray")
    pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    return torch.from_numpy(pairs).t().contiguous().to(torch.long)


def build_batch(frames, cutoff, device, dtype):
    """List of (z, R) -> AtomTrick batch dict with n_mol = len(frames)."""
    Z_list, R_list, N_list, edge_list, batch_idx = [], [], [], [], []
    offset = 0
    for i, (z, R) in enumerate(frames):
        n = len(z)
        edge_list.append(compute_edges(R, cutoff) + offset)
        Z_list.append(torch.from_numpy(z))
        R_list.append(torch.from_numpy(R))
        N_list.append(torch.tensor(n, dtype=torch.long))
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
        offset += n
    return {
        "Z":          torch.cat(Z_list).to(device),
        "R":          torch.cat(R_list).to(device=device, dtype=dtype),
        "N":          torch.stack(N_list).to(device),
        "edge_index": torch.cat(edge_list, dim=1).to(device),
        "batch":      torch.cat(batch_idx).to(device),
    }


# --------------------------------------------------------------------------- #
#  Timing
# --------------------------------------------------------------------------- #
def timed(fn, warmup, iters):
    for _ in range(warmup):        # the 1st makefx call compiles; latency EXCLUDES it
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(st, en)], torch.cuda.max_memory_allocated()


def summarize(times, bs):
    srt = sorted(times)
    n = len(srt)
    mean, median = statistics.mean(times), statistics.median(times)
    return {
        "latency_median_ms": median,
        "latency_mean_ms":   mean,
        "latency_p10_ms":    srt[max(0, int(0.1 * n) - 1)],
        "latency_p90_ms":    srt[min(n - 1, int(0.9 * n))],
        "latency_std_ms":    statistics.stdev(times) if n > 1 else 0.0,
        "throughput_mean_structs_per_s":   bs * 1000.0 / mean,
        "throughput_median_structs_per_s": bs * 1000.0 / median,
    }


FIELDS = [
    "dataset", "molecule", "atoms", "edges", "batch_size", "n_params", "path",
    "warmup", "iters", "latency_median_ms", "latency_mean_ms", "latency_p10_ms",
    "latency_p90_ms", "latency_std_ms", "throughput_mean_structs_per_s",
    "throughput_median_structs_per_s", "peak_mem_gb", "tf32", "weights",
    "frame_source", "gpu_name", "torch_version", "status",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=None,
                    help="YAML/JSON model config (repeatable; later override earlier). "
                         "Default: args.json next to --checkpoint if present, else "
                         "configs/rmd17/atomtrick.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="optional trained weights (*_best.pt raw EMA state_dict or a full "
                         "*_final.pt dict); random init otherwise (as reported)")
    ap.add_argument("--mode", choices=["makefx", "eager", "both"], default="both")
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--tf32", choices=["on", "off"], default="off",
                    help="off (default) = matmul precision 'highest' (full fp32; the release "
                         "default and what the baselines use). on = 'high' (TF32; not used "
                         "for any reported number)")
    ap.add_argument("--datasets", nargs="+", choices=["rmd17", "md22"],
                    default=["rmd17", "md22"])
    ap.add_argument("--molecules", nargs="+", default=None,
                    help="subset, e.g. aspirin Ac-Ala3-NHMe (default: all 17)")
    ap.add_argument("--frame_source", choices=["processed", "raw"], default="processed")
    ap.add_argument("--data_root", default="data",
                    help="processed-data root: <data_root>/<dataset>/<mol>_<split>_default.npz")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--raw_root", default=os.environ.get("ATOMTRICK_RAW_ROOT"),
                    help="for --frame_source raw: directory holding raw/rmd17/npz_data/*.npz "
                         "and raw/md22/*.npz (env ATOMTRICK_RAW_ROOT)")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="neighbour-list cutoff in A (default: the model's radial cutoff, 5.0; "
                         "any other value is rejected)")
    ap.add_argument("--seed", type=int, default=0, help="random-init seed")
    ap.add_argument("--output", default="benchmarks/latency/results/atomtrick_latency.csv")
    args = ap.parse_args()

    old = sorted(k for k in os.environ if k.startswith("ATOMTRICKS_"))
    if old:
        sys.exit(f"ERROR: old-spelling env vars {old} are set; the package was renamed "
                 "AtomTricks -> AtomTrick and these are ignored. Unset them / use ATOMTRICK_*.")
    if args.frame_source == "raw" and not args.raw_root:
        sys.exit("ERROR: --frame_source raw needs --raw_root (or env ATOMTRICK_RAW_ROOT)")
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA required for a GPU latency benchmark")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}  torch {torch.__version__}  CUDA {torch.version.cuda}", flush=True)

    # ---- model ----------------------------------------------------------------
    if args.config:
        config_paths = args.config
    elif args.checkpoint and os.path.exists(
            os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "args.json")):
        config_paths = [os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                                     "args.json")]
    else:
        config_paths = [DEFAULT_CONFIG]
    model, cfg, weights = build_model(config_paths, args.checkpoint, device, args.seed)
    # Set AFTER building: Trainer.from_config may set its own matmul precision.
    torch.set_float32_matmul_precision("high" if args.tf32 == "on" else "highest")
    dtype = next(model.parameters()).dtype
    cutoff = inference_cutoff(cfg, args.cutoff)   # the model's radial cutoff; must match
    n_params = sum(p.numel() for p in model.parameters())
    print(f"config: {config_paths}\nweights: {weights}  params: {n_params:,}  dtype: {dtype}  "
          f"tf32: {args.tf32}  cutoff: {cutoff} A  frames: {args.frame_source}\n", flush=True)

    specs = ([("rmd17", m) for m in RMD17 if "rmd17" in args.datasets]
             + [("md22", m) for m in MD22 if "md22" in args.datasets])
    if args.molecules is not None:
        keep = {_ALIASES.get(m, m) for m in args.molecules}
        specs = [(d, m) for (d, m) in specs if m in keep]
    if not specs:
        sys.exit("ERROR: no molecules selected")
    paths = ["eager", "makefx"] if args.mode == "both" else [args.mode]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    writer.writeheader()
    csv_file.flush()

    def emit(row):
        writer.writerow(row); csv_file.flush(); os.fsync(csv_file.fileno())

    def base_row(dataset, mol, na, ne, bs, path, status="ok"):
        row = {k: None for k in FIELDS}
        row.update({
            "dataset": dataset, "molecule": mol, "atoms": na, "edges": ne,
            "batch_size": bs, "n_params": n_params, "path": path,
            "warmup": args.warmup, "iters": args.iters, "tf32": args.tf32,
            "weights": weights, "frame_source": args.frame_source,
            "gpu_name": gpu_name, "torch_version": torch.__version__, "status": status,
        })
        return row

    def run_path(name, fn, dataset, mol, na, ne, bs):
        row = base_row(dataset, mol, na, ne, bs, name)
        try:
            times, peak = timed(fn, args.warmup, args.iters)
            row.update(summarize(times, bs))
            row["peak_mem_gb"] = peak / (1024 ** 3)
            print(f"    bs={bs} {name:>7}: median={row['latency_median_ms']:7.2f} ms  "
                  f"mean={row['latency_mean_ms']:7.2f}  p90={row['latency_p90_ms']:7.2f}  "
                  f"thr(mean)={row['throughput_mean_structs_per_s']:8.1f}/s  "
                  f"thr(median)={row['throughput_median_structs_per_s']:8.1f}/s  "
                  f"peak={row['peak_mem_gb']:5.2f} GB", flush=True)
        except RuntimeError as e:
            msg = str(e)
            row["status"] = "oom" if "out of memory" in msg.lower() else f"error: {msg[:100]}"
            print(f"    bs={bs} {name:>7}: {row['status']}", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            row["status"] = f"error: {str(e)[:100]}"
            print(f"    bs={bs} {name:>7}: ERROR {e}", flush=True)
            traceback.print_exc()
        emit(row)

    for bs in args.batch_sizes:
        print(f"\n########## BATCH SIZE {bs} ##########", flush=True)
        makefx_step = None
        if "makefx" in paths:
            # ONE dynamic graph per batch size (n_mol = bs is baked in), traced on the
            # first (smallest) molecule and reused for all others.  train_mode=True is the
            # create_graph=True trace -- the only one make_fx captures correctly (see
            # Trainer._makefx_infer_forward); it is called under no_grad below, so no
            # parameter backward is built.
            try:
                b0 = build_batch(load_frames(args, *specs[0], bs), cutoff, device, dtype)
                t0 = time.time()
                with torch.enable_grad():
                    makefx_step = MakeFXCompiledStep(model, b0, dynamic=True)
                with torch.no_grad():
                    pc = makefx_step.step(b0)
                pe = model(b0)
                dF = (pe["F"].detach() - pc["F"].detach()).abs().max().item()
                relF = dF / max(pe["F"].detach().abs().max().item(), 1e-8)
                print(f"  make_fx graph (bs={bs}): {makefx_step._n_nodes} nodes, "
                      f"n_mol={makefx_step._traced_nmol}, traced in {time.time() - t0:.1f}s "
                      f"(compile happens in warmup), relF_vs_eager={relF:.2e}", flush=True)
            except Exception as e:  # noqa: BLE001
                makefx_step = None
                print(f"  make_fx BUILD FAILED (bs={bs}): {e}", flush=True)
                traceback.print_exc()

        def makefx_fn(b):
            with torch.no_grad():
                return makefx_step.step(b)

        for dataset, mol in specs:
            try:
                frames = load_frames(args, dataset, mol, bs)
                batch = build_batch(frames, cutoff, device, dtype)
                na = len(frames[0][0])
                ne = batch["edge_index"].shape[1]
            except Exception as e:  # noqa: BLE001
                print(f"  {dataset}/{mol}: LOAD FAILED {e}", flush=True)
                for name in paths:
                    emit(base_row(dataset, mol, None, None, bs, name,
                                  status=f"load_error: {str(e)[:60]}"))
                continue
            print(f"  === {dataset}/{mol}: {na} atoms, {ne} edges (bs={bs}) ===", flush=True)
            if "eager" in paths:
                run_path("eager", lambda b=batch: model(b), dataset, mol, na, ne, bs)
            if "makefx" in paths:
                if makefx_step is not None:
                    run_path("makefx", lambda b=batch: makefx_fn(b), dataset, mol, na, ne, bs)
                else:
                    emit(base_row(dataset, mol, na, ne, bs, "makefx",
                                  status="skip: makefx build failed"))

    csv_file.close()
    print(f"\nCSV saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
