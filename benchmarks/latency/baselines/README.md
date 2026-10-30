# Baseline latency harnesses

The four scripts behind the baseline rows of the latency tables. Each is the baseline's own
benchmark script, extended with path arguments, the protocol switches below and
`--compile {none,inductor,fused}` (shared helpers in `latency_common.py`), and each runs
**inside a checkout of the baseline's own repository and environment**, not the ATOMTRICK
environment. `../run_all.sh` runs all of them; `../README.md` describes the protocol and the two
compile recipes.

**With default arguments every harness times the reported eager rows**: full fp32
(`--matmul-precision highest`), fp32 weights and inputs, real frames 0..bs−1 of the raw
trajectories (`--coords raw` for Equiformer and EquiformerV2; MACE and GotenNet always use
them), autograd forces with `create_graph=False` (`--create-graph false`, `--forces autograd`),
batch sizes 1 and 4, 10 warm-up + 50 timed calls. The reported compiled rows are the same
commands with `--compile inductor` and `--compile fused`. Every CSV row records the settings in
force, the graph size, the cost of an in-forward radius graph timed alone
(`graph_build_median_ms`), the parameter count and the torch / CUDA versions; compiled rows add
the `compile_*` columns. The other switches (`--coords synthetic|real`, `--forces direct`,
`--dtype float64`) are not used for any reported row.

| sub-dir | model | upstream repository @ commit | local changes |
|---|---|---|---|
| `equiformer/` | Equiformer (v1) | `https://github.com/atomicarchitects/equiformer` @ `64cb7866f48b9aa156e74a9d6a2ef2663b367437` | `nets_init.patch` drops the OC20 / dot-product-attention imports from `nets/__init__.py`; needs `../equiformer_v2/ocpmodels_shim/` on `PYTHONPATH` |
| `equiformer_v2/` | EquiformerV2 | `https://github.com/atomicarchitects/equiformer_v2` @ `d5ad4be729b56f74012ebb7f097f77c5b00a1004` | none; needs `ocpmodels_shim/` on `PYTHONPATH` (a stub of the few `ocpmodels` symbols it imports; `ocpmodels/models/scn/smearing.py` reproduces OCP's file, MIT, notice kept) |
| `gotennet/` | GotenNet | `https://github.com/sarpaykent/GotenNet` @ `16c7026b23584d13db3c30bc0c80cecece775c1a` (branch `rMD17`) | `gotennet_local.patch`: an MD22 dataset class and configs, the `rmd17_aspirin` experiment with n_interactions 16 → 12; `gotennet_compile.patch`: three in-place operations rewritten out of place (required by `torch.compile`, applied for the eager rows too; outputs agree to ~1e-6) |
| `mace/` | MACE | `https://github.com/ACEsuit/mace` @ `667eee4e58d23a38ff5a75122109ec2025809649` (v0.3.15 + 46 commits) | none; the compiled rows need the same package in a Python ≤ 3.12 environment |

## Per-model configuration

| model | configuration | parameters | graph construction | compiled recipe shown |
|---|---|---|---|---|
| Equiformer | MD17 `nonlinear_exp_l2`: 6 layers, l_max 2, `128x0e+64x1e+32x2e`, 4 heads, r_c 5 Å | 3,496,001 | `radius_graph` inside `forward` (**timed**) | `inductor` (`fused` runs out of memory) |
| EquiformerV2 | OC20 `N@12_L@6_M@2`: 12 layers, l_max 6, m_max 2, 128 sphere channels, max 50 neighbours; direct force head not used | 77,182,465 | on-the-fly graph inside the model (**timed**) | `inductor` (`fused` runs out of memory) |
| GotenNet | rMD17: `experiment=rmd17_aspirin` (12 layers, 192 ch, l_max 2); MD22: `experiment=md22_at_at` (8 layers, 384 ch, l_max 2) | 21,069,057 / 42,603,777 | `radius_graph` inside the model (**timed**) | `inductor` |
| MACE | ScaleShiftMACE "256-2": 2 interactions, `256x0e+256x1o+256x2e`, max_ell 3, correlation 3, r_max 5 Å | 3,769,616 | in `build_batch` (not timed) | `fused` (one graph) |
| ATOMTRICK (`../bench_atomtrick.py`) | paper model | 3,730,368 | cKDTree outside the timed region (not timed) | make_fx + `torch.compile` |

Caveats to state next to any comparison:

1. Graph construction is inside the timed region for Equiformer, EquiformerV2 and GotenNet
   (0.2–0.3 ms per call, `graph_build_median_ms`), eager and compiled alike; it is outside for
   ATOMTRICK and MACE.
2. Compare compiled with compiled: compiling helps every model, ATOMTRICK's make_fx path most.
   Only MACE's `fused` recipe reaches one graph, and compiled MACE runs in a Python 3.11 copy of
   its environment (eager MACE in the original Python 3.13 environment agreed to within ±3 %).
3. EquiformerV2 is an OC20 model never trained on these data, timed with autograd forces: an
   architecture-cost reference only.
4. GotenNet uses a different configuration per dataset; parameter ratios must say which one.

## Environments

Each baseline ran in its own conda environment built from its upstream instructions (Equiformer
in the EquiformerV2 environment): torch 2.5.1 everywhere, Python 3.10 with cu124 wheels for
Equiformer / EquiformerV2 / GotenNet, Python 3.11 with cu121 for MACE and ATOMTRICK. A
`pip freeze` of every environment is in `../results/h100_fp32_2026-10-20/env/`.

## Commands

`DATA` holds `raw/rmd17/npz_data/rmd17_<mol>.npz` and `raw/md22/md22_<mol>.npz`. Output names
follow `merge.py`'s convention: `<prefix>_<dataset>_latency_<gpu>.csv` for eager rows,
`<prefix>_compiled_<variant>_<dataset>_latency_<gpu>.csv` for compiled rows. All four harnesses
need a CUDA GPU; `--compile` also needs a working Triton (Python ≤ 3.12 with the torch 2.5.1
wheels).

```bash
H=$PWD/benchmarks/latency/baselines          # run from the ATOMTRICK repo root
OUT=$PWD/benchmarks/latency/results/raw_new; mkdir -p $OUT
# compiled rows: add --compile inductor (or fused) and name the output <model>_compiled_<variant>_${ds}_latency_h100.csv

# --- Equiformer (inside the equiformer_v2 environment) ---
git clone https://github.com/atomicarchitects/equiformer && git -C equiformer checkout 64cb786
git -C equiformer apply $H/equiformer/nets_init.patch
for ds in rmd17 md22; do
  PYTHONPATH=$H/equiformer_v2/ocpmodels_shim python $H/equiformer/benchmark_inference.py \
    --repo-root equiformer --raw-root $DATA --dataset $ds --batch-sizes 1 4 --warmup 10 --iters 50 \
    --output $OUT/equiformer_${ds}_latency_h100.csv
done

# --- EquiformerV2 ---
git clone https://github.com/atomicarchitects/equiformer_v2 && git -C equiformer_v2 checkout d5ad4be
for ds in rmd17 md22; do
  PYTHONPATH=$H/equiformer_v2/ocpmodels_shim python $H/equiformer_v2/benchmark_inference.py \
    --repo-root equiformer_v2 --raw-root $DATA --dataset $ds --batch-sizes 1 4 --warmup 10 --iters 50 \
    --output $OUT/equiformer_v2_${ds}_latency_h100.csv
done

# --- GotenNet ---
git clone https://github.com/sarpaykent/GotenNet && git -C GotenNet checkout 16c7026
git -C GotenNet apply $H/gotennet/gotennet_local.patch
git -C GotenNet apply $H/gotennet/gotennet_compile.patch
pip install -e GotenNet
for ds in rmd17 md22; do
  PROJECT_ROOT=$PWD/GotenNet python $H/gotennet/benchmark_inference.py \
    --repo-root GotenNet --data-dir $DATA --dataset $ds --batch-sizes 1 4 \
    --warmup 10 --iters 50 --output $OUT/gotennet_${ds}_latency_h100.csv
done

# --- MACE (compiled rows: a Python <= 3.12 environment with the same mace / torch 2.5.1) ---
git clone https://github.com/ACEsuit/mace && git -C mace checkout 667eee4 && pip install -e mace
for ds in rmd17 md22; do
  python $H/mace/benchmark_inference_mace.py --data-root $DATA --dataset $ds \
    --batch-sizes 1 4 --warmup 10 --iters 50 --output $OUT/mace_${ds}_latency_h100.csv
done

# --- merge with an ATOMTRICK run (eager and compiled baseline files are picked up by name) ---
python benchmarks/latency/merge.py --baseline_dir $OUT --gpu_tag h100 --compiled_variant auto \
  --atomtrick_csv benchmarks/latency/results/atomtrick_latency.csv \
  --out_long $OUT/latency_all_long.csv --out_md $OUT/latency_all_summary.md
```
