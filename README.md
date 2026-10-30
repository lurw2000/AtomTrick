# ATOMTRICK: Equivariant Force Fields via Learned O(2)-Linear Projections

Code for the paper *"ATOMTRICK: Equivariant Force Fields via Learned
O(2)-Linear Projections"*. The Python package is `AtomTrick`.

ATOMTRICK is an O(3)-equivariant machine-learned force field that does not use
Clebsch–Gordan tensor products.

**Edge projections.** For every edge, the neighbour's features are rotated into the
bond-aligned frame. In that frame a learned linear map mixes channels and angular
degrees within each |m| sector, using the same weights for ±m, so the map is
O(2)-linear. The result is rotated back and scaled by a radial gate: one value per
(degree, channel), produced by a bias-free MLP on an exponential-normal radial basis
with a cosine cutoff.

**Model.** The O(2)-linear maps produce the keys and values of an equivariant
multi-head attention with **unnormalised (identity) attention weights**
⟨q_i, k_ij⟩/√d. Each of the **6 pre-norm layers** uses a **shared equivariant RMS
LayerNorm** (one RMS over all degrees) and a **gated equivariant feed-forward
network**. A final LayerNorm feeds a **scalar readout**: an MLP on the l = 0 channels
gives per-atom energies, which are summed. Forces are **F = −∂E/∂R by autograd**.

**Configuration.** Irreps `64x0e+64x1o+64x2e+64x3o`, 8 heads, **3,730,368 parameters**.
Training and inference run through a compiled path: `make_fx` traces the energy *and*
the force backward into one graph, which `torch.compile` compiles
(`AtomTrick/backward/makefx_compiled.py`). There are no hand-written kernels. Every reported model is trained and evaluated in full fp32
(matmul precision `highest`, no TF32) with float64 label arithmetic.

---

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Reproducing the paper](#reproducing-the-paper)

## Repository layout

```
AtomTrick/            the package: model, attention, O(3)/O(2) primitives (o3/), trainer, data
train.py              training entry point (YAML configs + CLI overrides, --dry_run)
evaluate.py           test-set evaluation of one or many checkpoints -> JSON / CSV
configs/              one YAML per paper experiment (see configs/README.md)
scripts/data/         download / prepare / verify the datasets
scripts/reproduce/    bash drivers for every experiment
scripts/aggregate/    evaluation JSON -> paper tables (mean ± std over seeds, baselines)
benchmarks/latency/   inference latency / throughput / memory benchmark
md/                   OpenMM molecular dynamics (NVE, NVT, Ramachandran)
tests/                pytest suite (CPU; GPU-only tests skip automatically)
LICENSE               MIT license of the code (Copyright (c) 2026 The ATOMTRICK Authors);
                      THIRD_PARTY_NOTICES.md lists the adapted components
```

## Installation

Every reported run used a single NVIDIA GPU and Python 3.11 with the pinned versions
below (PyTorch 2.5.1 with CUDA 12.1 wheels, Triton 3.1.0, ...). Each run records its
versions in `args.json` under `_provenance`.

```bash
git clone https://github.com/lurw2000/AtomTrick.git && cd AtomTrick

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # exact versions
pip install -e .                  # installs the AtomTrick package; add [wandb], [md], [dev] as needed
```

- `pyproject.toml` lists the unpinned dependencies: torch ≥ 2.5, numpy, scipy, ase,
  pandas, pyyaml, tqdm. Extras: `wandb`; `md` (OpenMM 8.5.2, matplotlib); `dev`
  (pytest).
- Weights & Biases logging is off unless you pass `--wandb_project`.
- The compiled path compiles once per run, which takes a few minutes. Point
  `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` at a fast local disk.

## Data

All commands run from the repository root. Raw downloads go to `data/raw/`, processed files to
`data/<dataset>/`; both are git-ignored. `scripts/data/README.md` has the details: sources,
licences, split construction, checksums.

```bash
bash scripts/data/download_rmd17.sh          && python scripts/data/prepare_rmd17.py          # 1.1 GB
bash scripts/data/download_md22.sh           && python scripts/data/prepare_md22.py           # 565 MB
bash scripts/data/download_spice_maceoff.sh  && python scripts/data/prepare_spice_maceoff.py  # 1.7 GB
python scripts/data/verify_data.py           # array-level checksums against the files we trained on
```

The processed files are `data/rmd17/<molecule>_{train,val,test}_default.npz`,
`data/md22/<system>_{train,val,test}_default.npz` and
`data/spice_maceoff/spice_maceoff_{train,val,test,test_f15}.npz`.

| Dataset | Split |
|---|---|
| rMD17: aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic, toluene, uracil | official split 1: 950 train / 50 validation (from the training indices, seed 0) / 1,000 test |
| MD22: Ac-Ala3-NHMe, DHA, AT-AT, stachyose, AT-AT-CG-CG, buckyball-catcher, double-walled_nanotube | random, seed 1; training sizes of Chmiela et al. (2023); 500 validation (100 for the two largest); the rest is test |
| SPICE (MACE-OFF23 release) | MACE-OFF23 train file with 5 % held out for validation; MACE-OFF23 test file (50,195) and its force-filtered copy `test_f15` (50,069); results are reported on `test_f15` only |

NPZ keys:

- `N, Z, R, F, E`, plus `subset` for SPICE.
- `E` is the total energy in kcal/mol; `F` is in kcal/mol/Å.
- The loader converts to eV and subtracts atomic reference energies, in float64.

## Quick start

```bash
# Inspect the fully resolved configuration and the parameter count (no data, no GPU)
python train.py --config configs/rmd17/atomtrick.yaml --molecule aspirin --dry_run
#   ... Model parameters: 3,730,368

# Train the paper model on rMD17 aspirin (~9 h on one L40S)
python train.py --config configs/rmd17/atomtrick.yaml --molecule aspirin --seed 42

# Evaluate its best (EMA) checkpoint on the 1,000-structure test split
python evaluate.py \
    --checkpoint "runs/rmd17_atomtrick/{molecule}/s42/rmd17_atomtrick_{molecule}_s42_best.pt" \
    --data "data/rmd17/{molecule}_test_default.npz" --molecules aspirin \
    --out results/rmd17_aspirin_s42.json
```

What a run produces:

- The model settings and paths come from the config file. `{molecule}`, `{seed}` and
  `{experiment}` are filled in; the rMD17 run above writes to
  `runs/rmd17_atomtrick/aspirin/s42/`.
- Each run writes `args.json` (the resolved configuration plus a `_provenance`
  record), `<name>_best.pt` (the EMA weights at the lowest validation loss; this is
  the checkpoint that is evaluated) and `<name>_final.pt` (the full training state).
- `evaluate.py` reads the model configuration from the `args.json` next to the
  checkpoint.
- `evaluate.py` reports energy MAE/RMSE (total and per atom), force MAE/RMSE per
  component, and force L2 (mean over atoms of ‖ΔF_i‖₂). All are pooled over the whole
  set in float64, in meV and meV/Å.
- Add `--per_subset` for SPICE.

Compiled path: `train.py --makefx auto` (the default) trains through make_fx +
`torch.compile` whenever CUDA is available; `--makefx off` forces eager autograd, which is
slower but numerically equivalent. `evaluate.py` is eager by default (`--makefx off`);
`--makefx on` compiles an inference graph, which pays off only on large test sets.

## Configuration

- **Defaults.** `train.py`'s defaults *are* the paper configuration, including
  the rMD17/MD22 training recipe: AdamW, lr 2e-4, weight decay 0, batch 4, 750k steps,
  ρ_F = 0.999, EMA 0.99, 10k warmup steps, ReduceLROnPlateau.
- **Config files.** Keys are exactly the option names (`n_layers: 6`,
  `attn_nonlin: identity`, `k_mode: o2`, ...). An unknown key is an error.
- **Merge order.** `--config` may be repeated and later files override earlier ones.
  Explicit command-line flags override every file. For example,
  `--config configs/rmd17/atomtrick.yaml --config configs/rmd17/ablations/so2.yaml`
  is the SO(2) ablation.
- **Retired options.** A few option names that no longer do anything (`manual_backward`,
  `cuda_graph`, `compile`, `radial_rank`, `energy_correlation`, ...) are accepted and
  ignored with a warning, so that an `args.json` written by an older version of this
  code still resolves.
- **Environment variables.**
  - `train.py` sets `PYTORCH_JIT=0` and the `ATOMTRICK_MAKEFX*` variables itself.
  - It refuses to start if any `ATOMTRICKS_*` variable (the spelling before the
    package was renamed) is set, because those are silently ignored.
- **Matmul precision.** `--matmul_precision highest` (the default, also in every config)
  keeps full fp32 matmuls; `high` enables TF32 matmuls on Ampere and newer GPUs. The value
  is recorded in `args.json`, and `evaluate.py` scores each checkpoint with the precision it
  was trained with (an `args.json` that records none is scored with `high`);
  `evaluate.py --matmul_precision` overrides it. With a YAML `evaluate.py --config` the
  YAML's `matmul_precision` applies instead.
- **eSCN frame.** `--escn_frame polefree` (the default) gives exactly equivariant
  forces for every edge direction; `euler` is the direct Euler-angle evaluation of the
  same frame, the same function up to round-off, kept for the comparison of the two frames
  in the paper's latency appendix. `evaluate.py --escn_frame` overrides it.

`configs/README.md` maps every config file to the paper result it produces.

## Reproducing the paper

Every experiment has a driver in `scripts/reproduce/`. Each one trains all runs of the
experiment (molecules × seeds) and then scores every run's `*_best.pt` on its test
split. It writes `eval_test.json` (SPICE: `eval_test_f15.json`) into the run directory.

- **Plain bash.** The scripts run locally, one run after another; split the work over several
  jobs with `--molecules` / `--seeds`.
- **`--dry`** prints the commands without running them.
- **Restarts.** Finished runs are skipped and interrupted runs resume from their
  rolling checkpoint, so re-running a script after a timeout is safe.
- **Tables.** `bash scripts/reproduce/eval_all.sh` then evaluates anything not yet
  evaluated and builds the tables in `results/`. It uses
  `scripts/aggregate/{rmd17_table,md22_table,spice_table}.py`, which report mean ±
  sample std over seeds, win counts and relative gaps against the published baselines
  in `scripts/aggregate/baselines.py`.

Configs are relative to `configs/`, drivers to `scripts/reproduce/`. *Hardware / time* is what the
reported runs used (measured wall time per training run; per-variant and per-system times are in
the config headers).

| Paper result | Config → driver | Hardware / time |
|---|---|---|
| **rMD17 main table** (energy MAE, force MAE, force L2; 10 molecules × seeds 42, 1, 2) | `rmd17/atomtrick.yaml` → `rmd17.sh` | 1× L40S, 9–13 h per run; 305 GPU-h in total |
| **O(2) vs SO(2) K/V** (the main ablation; 3 seeds; the SO(2) model has 18 % more parameters) | + `rmd17/ablations/so2.yaml` → `rmd17_ablations.sh so2` | 1× L40S, 11–16 h per run; 337 GPU-h in total |
| **Other rMD17 ablations** (softmax attention, per-degree LayerNorm, O(3)-linear K/V, L_max 1 / 2, 4 / 8 / 12 layers; seed 42) | + `rmd17/ablations/<name>.yaml` → `rmd17_ablations.sh <name>` (or `all`) | 1× L40S, 6–17 h per run depending on the variant |
| **MD22**, 7 systems (energy and force MAE, force L2) | `md22/atomtrick.yaml` → `md22.sh` | 1× H100 for the 3 small systems (≈ 11 h each), 1× H200 for the 4 large ones (17–64 h); ≈ 155 GPU-h in total |
| **SPICE vs MACE-OFF23** (7 subsets; force-filtered test set only) | `spice/stage1_force.yaml` → + `spice/stage2_swa.yaml` → `spice.sh` | 1× H200, ≈ 102 GPU-h |
| **Latency, throughput, memory** (batch sizes 1 and 4; all models full fp32, baselines eager and compiled) | `benchmarks/latency/run_all.sh` (see its README) | 1× H100, ≈ 1 h 40 min |
| **OpenMM MD** on Ac-Ala3-NHMe (NVE drift, 300 K NVT, Ramachandran) | `md/md_openmm.py` with the MD22 Ac-Ala3-NHMe checkpoint (see `md/README.md`) | 1× L40S, ≈ 5 h per 1 ns run |
| **Parameter count** (3,730,368) | `python train.py --config … --dry_run` | CPU, seconds |

**Reference results.** The reported numbers are in the paper; no trained checkpoints are
released. Every run is full fp32 (`matmul_precision: highest`,
float64 labels, AdamW), scored on the test sets. Runs are not bit-deterministic (CUDA
`scatter_add` atomics and Inductor kernel selection change the summation order; across seeds,
rMD17 force MAE varies by about 3 %), so a re-run reproduces the numbers statistically, not bit
for bit.

**Single runs by hand.** The scripts are thin loops around these two commands:

```bash
python train.py --config configs/rmd17/atomtrick.yaml [--config configs/rmd17/ablations/<name>.yaml] \
    --molecule <mol> --seed <seed>
python evaluate.py --checkpoint "runs/<experiment>/{molecule}/s<seed>/<experiment>_{molecule}_s<seed>_best.pt" \
    --data "data/rmd17/{molecule}_test_default.npz" \
    --molecules aspirin azobenzene benzene ethanol malonaldehyde naphthalene paracetamol salicylic toluene uracil \
    --out results/<experiment>_s<seed>.json
```

Here `<experiment>` is the `experiment:` value of the last config file, e.g.
`rmd17_atomtrick` or `rmd17_so2`.

**Scoring a directory of checkpoints.** Any directory in the `runs/` layout
(`<experiment>/<molecule>/s<seed>/` with the checkpoint and its `args.json`; SPICE
`<experiment>/s<seed>/`; experiment names start with `rmd17_`, `md22_` or `spice_`), for
example the output of the drivers above on another machine, works with the evaluation driver
and the aggregators directly:

```bash
bash scripts/reproduce/eval_all.sh --runs <runs_dir> --dry     # list what would be scored
bash scripts/reproduce/eval_all.sh --runs <runs_dir>
```

Runs trained with this code use float64 labels and record their matmul precision in
`args.json`, so no extra flags are needed. `RESULTS_DIR` (default `results/`) receives the
tables.
