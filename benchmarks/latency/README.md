# Inference latency benchmarks

GPU latency, throughput and peak memory of energy + force inference: ATOMTRICK against
Equiformer, EquiformerV2, GotenNet and MACE on the 10 rMD17 and 7 MD22 systems at batch sizes 1
and 4, every model in full fp32, every baseline eager and under `torch.compile`.

```
bench_atomtrick.py      ATOMTRICK harness (eager and make_fx + torch.compile)
baselines/              the four baseline harnesses, patches and provenance (see baselines/README.md)
run_all.sh              driver: all five models, one GPU, one session (the reported measurement)
merge.py                per-model CSVs -> one long CSV + markdown tables (picks the compiled variant per baseline)
plot.py                 latency-vs-atoms figure from the long CSV
conditions.py           recorded measurement conditions + consistency checks for a run_all.sh run
env_probe.py            per-environment version probe used by run_all.sh
escn_euler.yaml         config override for the Euler-edge-frame extra
results/h100_fp32_2026-10-20/   the reported measurement (one H100 80GB, 2026-10-20)
```

## Protocol

Every model in strict fp32 (`torch.set_float32_matmul_precision('highest')`), random-init
weights, frames 0..bs−1 of the raw rMD17 / MD22 trajectories, energy + forces by autograd,
10 warm-up + 50 timed calls with per-call `torch.cuda.Event` timing, median latency; throughput
is `bs·1000/median`.

**ATOMTRICK compiled** is the training path (`AtomTrick/backward/makefx_compiled.py`): `make_fx`
traces the forward pass and the force backward into one graph, which `torch.compile(dynamic=True)`
compiles; one graph per batch size is reused across molecules, the warm-up absorbs trace and
compile time, and the compiled forces are checked against eager (relative error of order 1e-6).
The **eager** rows are plain autograd, several times slower. ATOMTRICK's advantage over the
*eager* baselines comes from compiling the force graph, so the paper compares compiled models
with compiled baselines.

**Baselines compiled**: `--compile {inductor,fused}` wraps each baseline's whole energy + force
callable in `torch.compile(dynamic=True)`. `inductor` graph-breaks at `autograd.grad` and takes
the forces from the compiled backward; `fused` inlines it, so forward and force backward form one
graph (MACE's upstream recipe, generalised). Only MACE reaches one graph; the PyG models
graph-break at their `torch_cluster` radius graph, which therefore still runs eagerly inside the
timed region. `merge.py --compiled_variant auto` shows per baseline the faster recipe (MACE:
`fused`; the others: `inductor`); both are in the long CSV. Compiled rows record warm-up seconds,
graph counts and the max relative error against one eager call; a failure is a `compile_error`
row, never a silent eager fallback. The torch 2.5.1 workarounds (opaque `torch_cluster` ops,
`specialize_int`, single-threaded autograd, EquiformerV2's cache helpers outside the graph,
`gotennet_compile.patch`, MACE in a Python 3.11 environment) are documented in
`baselines/latency_common.py` and the harness docstrings.

## The reported measurement (`results/h100_fp32_2026-10-20/`)

One NVIDIA H100 80GB HBM3 (host AMD EPYC 9454, 8 cores), all models in one session. The paper's
two efficiency tables are the compiled columns of the batch-1 and batch-4 latency and memory
tables, its MD22 figure the batch-4 MD22 rows.

| file | content |
|---|---|
| `latency_all_summary_d5.md` | the tables: latency (eager and compiled), peak memory, throughput, parameter counts, the compiled-variant table, the recorded conditions |
| `MEASUREMENT_CONDITIONS.md`, `conditions.json` | `conditions.py` output: recorded protocol, versions per environment, graph sizes, compiled-baseline statistics, the Euler-frame ratio, consistency checks |
| `latency_all_long.csv` | one row per dataset × model × molecule × batch size × recipe (`selected` marks the recipe shown) |
| `atomtrick_latency.csv`, `raw/*.csv` | per-model harness outputs: eager and both compiled recipes per baseline |
| `extra/atomtrick_euler_latency.csv` | ATOMTRICK with the direct Euler-angle edge frame (`escn_frame: euler`), for the pole-free vs Euler ratio quoted in the paper's latency appendix |
| `env/`, `run_meta.txt` | `env_probe.py` JSON and `pip freeze` per environment; code commits, harness checksums, hardware |
| `latency_all_summary.md` | `merge.py`'s raw output, the input of `conditions.py` |

The compiled-MACE `fused` MD22 step was re-run after an intermittent illegal-memory-access at
AT_AT_CG_CG, batch 4 (an Inductor-emitted `bmm` on torch 2.5.1; the rows before the failure agree
with the passing run). `conditions.py` records that re-run as a break of its one-session rule and
exits 1 on this set by design; its four remaining warnings are benign (graph breaks and
no-speedup rows). Host names and absolute paths were removed from `run_meta.txt` and
`env/*.json`; `merge.py` and `conditions.py` regenerate the committed summaries byte-for-byte
from the committed CSVs.

## Re-running it

`run_all.sh` runs every model in its own environment, sequentially, on the first GPU of
`CUDA_VISIBLE_DEVICES`: ATOMTRICK eager + compiled, the four baselines eager and compiled, the
Euler-frame extra, then merge, plot and the conditions report (about 1 h 40 min on an H100). Run
it from the repository root, outside any single model's environment, with the baseline clones set
up as in `baselines/README.md`.

```bash
export RAW_ROOT=/data/mlff          # raw/rmd17/npz_data/rmd17_<mol>.npz, raw/md22/md22_<mol>.npz
export BASELINES_DIR=/src/baselines # equiformer/ equiformer_v2/ GotenNet/ mace/
# how to enter each environment: <M>_PREFIX (command prefix) and <M>_ACTIVATE (snippet run inside it),
# M in ATOMTRICK MACE GOTENNET EQUIFORMER_V2 (EQUIFORMER defaults to EQUIFORMER_V2), e.g. conda:
export ATOMTRICK_ACTIVATE="conda activate atomtrick" MACE_ACTIVATE="conda activate mace" \
       GOTENNET_ACTIVATE="conda activate gotennet" EQUIFORMER_V2_ACTIVATE="conda activate equiformer_v2"
# or a container: MACE_PREFIX="singularity exec --nv --overlay /envs/mace.ext3:ro /envs/cuda12.sif" MACE_ACTIVATE="source /ext3/env.sh"
# compiled MACE needs a Python <= 3.12 copy of the MACE environment (torch 2.5.1 has no torch.compile on 3.13)
export MACE_COMPILED_ACTIVATE="conda activate mace311"   # and MACE_COMPILED_PREFIX if needed
DRY=1 bash benchmarks/latency/run_all.sh   # print every command, run nothing
OUT=benchmarks/latency/results/my_run GPU_TAG=h100 bash benchmarks/latency/run_all.sh
```

The knobs (`COMPILE_VARIANTS`, `COMPILED_VARIANT`, `GPU_TAG`, `ONLY`, `FORCE`, `EXTRAS`, the
molecule lists for a quick smoke) are documented in the header of `run_all.sh`. Finished CSVs are
skipped, so the driver is resumable; it exits non-zero if a step failed or `conditions.py` found
an inconsistency. Read `OUT/latency_all_summary_d5.md`. The baseline commands are in
`baselines/README.md`; the ATOMTRICK one is

```bash
python benchmarks/latency/bench_atomtrick.py --batch_sizes 1 4 --mode both --tf32 off \
  --frame_source raw --raw_root $RAW_ROOT --warmup 10 --iters 50 \
  --config configs/rmd17/atomtrick.yaml --output $OUT/atomtrick_latency.csv
```

`bench_atomtrick.py` defaults to `--frame_source processed` (test-split frames from `data/`); the
reported rows used `--frame_source raw`, the frames every baseline sees. `--tf32 off` (the
default) gives full fp32 matmuls. `--checkpoint <run>/<name>_best.pt` times trained weights
instead of random ones (latency does not depend on the weights).
