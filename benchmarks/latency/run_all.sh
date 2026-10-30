#!/bin/bash
# Inference latency of ALL models (ATOMTRICK eager + compiled, MACE, GotenNet, Equiformer,
# EquiformerV2, each baseline eager AND compiled) on ONE GPU in ONE session, full fp32 -- the
# driver behind the reported numbers (benchmarks/latency/results/h100_fp32_2026-10-20/, baselines
# eager and compiled, see benchmarks/latency/README.md).
#
# Run it on the GPU machine itself, NOT inside any one model's environment: each model runs in
# its own environment, sequentially, on the same GPU:
#
#   1 ATOMTRICK    eager + compiled (make_fx + torch.compile)   bench_atomtrick.py
#   2 MACE         eager                                         baselines/mace
#   3 GotenNet     eager                                         baselines/gotennet
#   4 Equiformer   eager                                         baselines/equiformer
#   5 EquiformerV2 eager                                         baselines/equiformer_v2
#   6 MACE, GotenNet, Equiformer, EquiformerV2 compiled: the same harness commands plus
#     --compile <variant> for each variant in COMPILE_VARIANTS (default "inductor fused"):
#       inductor  torch.compile(dynamic=True) of the whole energy + force callable; Dynamo
#                 graph-breaks at autograd.grad, forces from the compiled backward
#                 (create_graph=False as eager)
#       fused     the same with autograd.grad allowed in the graph: forward + force backward
#                 in ONE compiled graph (MACE's upstream torch.compile recipe, generalised;
#                 create_graph=True inside)
#     parameters frozen, PYTORCH_JIT=0 (set by the harness itself), one environment launch per
#     model and variant (both datasets), output raw/<model>_compiled_<variant>_<ds>_latency_<GPU_TAG>.csv.
#     Steps mace_compiled gotennet_compiled equiformer_compiled equiformer_v2_compiled.
#   7 extra (not in the merged table): ATOMTRICK with the direct Euler-angle evaluation of the
#     edge frame (escn_frame: euler; the pole-free vs Euler ratio quoted in the paper's latency
#     appendix)
#   8 merge.py + plot.py, then conditions.py (provenance + consistency checks)
#
# Time budget: the compiled steps add Dynamo trace + Inductor compile per new graph, which can
# take minutes per model and variant (more with dynamic-shape recompiles); it lands in the 10
# warm-up calls and is excluded from the latency, but it is wall-clock in the job.  A
# compile_error or oom row still counts as a row (no eager fallback), so a failing recipe does
# not stop the session; finalize logs a WARN line when every row of a file is compile_error.
# Expected: GotenNet and EquiformerV2 build their radius graph with a torch_cluster custom op
# (no fake kernel) inside the model, so their `fused` rows may all be compile_error (the graph
# break inside an inlined call keeps the inlined autograd.grad from reaching pos); the
# `inductor` rows are unaffected.  COMPILE_VARIANTS=inductor shortens the job accordingly.
# After the warm-up every compiled row's energy + forces are checked against one eager call of
# the same model (compile_max_rel_err column; a mismatch is a compile_error).
# A failing environment activation (M_ACTIVATE) aborts the launch (exit 97, "no output" FAIL
# rows) instead of running the harness in the outer environment.
#
# Common settings: strict fp32 (matmul precision 'highest' for every model; the baseline
# harnesses also switch cuDNN TF32 off, bench_atomtrick.py leaves it at PyTorch's default, which
# affects only cuDNN convolutions and ATOMTRICK has none), fp32 weights and inputs, random-init
# weights, frames 0..bs-1 of the raw rMD17 / MD22 trajectories for every model, energy + autograd
# forces with create_graph=False, bs 1 and 4, 10 warmup + 50 CUDA-event-timed calls, median
# reported.  The baseline harnesses default to these settings; bench_atomtrick.py defaults to
# --frame_source processed (test-split frames from data/) and needs --frame_source raw
# --raw_root, as passed below.  Every flag is spelled out here.
#
# GPU: every model times cuda:0, i.e. the first entry of CUDA_VISIBLE_DEVICES (device 0 if it is
# unset).  On a multi-GPU machine pick the GPU there (an index or a GPU-... UUID); the driver
# exports CUDA_DEVICE_ORDER=PCI_BUS_ID so an index means the same GPU to CUDA and to nvidia-smi,
# and restricts every nvidia-smi query (run_meta.txt, busy check, clocks, name) to that GPU.
#
# usage:  RAW_ROOT=... BASELINES_DIR=... [knobs] bash benchmarks/latency/run_all.sh
#         DRY=1 bash benchmarks/latency/run_all.sh     print every command, run nothing
#
# Required:
#   RAW_ROOT        directory with raw/rmd17/npz_data/rmd17_<mol>.npz and raw/md22/md22_<mol>.npz
#                   (GotenNet writes its processed .pt cache under it on first use: keep it writable)
#   BASELINES_DIR   directory holding the four baseline clones as equiformer/, equiformer_v2/,
#                   GotenNet/ and mace/ (set up as in baselines/README.md), or set each one:
#                   EQUIFORMER_ROOT, EQUIFORMER_V2_ROOT, GOTENNET_ROOT, MACE_ROOT.
#                   The GotenNet clone must carry baselines/gotennet/gotennet_compile.patch
#                   (three in-place operations made out of place, same values; without it
#                   torch.compile rejects the model); it is used by the eager rows as well.
# Environments: for M in ATOMTRICK MACE GOTENNET EQUIFORMER EQUIFORMER_V2,
#   M_PREFIX        command prefix that enters M's environment (default: none = this shell), e.g.
#                   "singularity exec --nv --overlay /path/mace.ext3:ro /path/image.sif"
#   M_ACTIVATE      shell snippet run first inside it, e.g. "conda activate mace" or
#                   "source /ext3/env.sh && conda activate mace" (default: none)
#   The command run is:  $M_PREFIX /bin/bash -lc "$M_ACTIVATE || exit 97; <python ...>"
#   (a failed activation ends the launch; nothing runs in the outer environment).
#   EQUIFORMER_* default to EQUIFORMER_V2_* (the reported Equiformer v1 rows ran in the
#   EquiformerV2 environment).  No word in a prefix may contain spaces.
#   MACE_COMPILED_PREFIX / MACE_COMPILED_ACTIVATE: the environment for the compiled MACE step
#   (default: MACE_PREFIX / MACE_ACTIVATE).  torch 2.5.1 has no torch.compile on Python 3.13,
#   so a Python 3.13 MACE environment cannot run it: point these at a Python <= 3.12 copy of it
#   (same torch 2.5.1, e3nn, mace; e.g. "source /ext3/env.sh && conda activate mace311").  Its
#   env_probe json goes to $OUT/env/mace_compiled.json (+ pip freeze); the other compiled steps
#   reuse their eager environment and json.
# Optional:
#   OUT=benchmarks/latency/results/local_run   results directory (resumable, see below)
#   GPU_TAG=h100    file-name tag (<model>_<dataset>_latency_<GPU_TAG>.csv) and merge.py --gpu_tag
#   TITLE_GPU       GPU name in the summary title (default: nvidia-smi name + ", full fp32")
#   COMPILE_VARIANTS="inductor fused"   compile recipes of step 6 (any subset, e.g. "fused")
#   COMPILED_VARIANT=auto               which variant merge.py shows as "<model> (compiled)"
#                                       (auto = the faster per model | inductor | fused)
#   EXTRAS=0        skip step 7;  ONLY="atomtrick mace ..." run a subset of the steps
#                   (atomtrick mace gotennet equiformer equiformer_v2 mace_compiled
#                   gotennet_compiled equiformer_compiled equiformer_v2_compiled extras merge)
#   FORCE=1         re-run steps whose CSV already exists
#   RMD17_MOLS="ethanol aspirin" MD22_MOLS="Ac_Ala3_NHMe stachyose"   molecule subsets (harness
#                   --molecules names; default all 10 / all 7) for a quick smoke of the whole
#                   pipeline; the expected row counts follow.  Not for reported numbers.
#
# Resumable: every finished CSV is validated (row count) and moved into place; a re-run skips it.
# Use FORCE=1 (or a fresh OUT) when the re-run lands on a different GPU: the comparison is only
# meaningful within one session, and conditions.py flags results that come from several GPUs.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
HARNESS=$HERE/baselines
SHIM=$HARNESS/equiformer_v2/ocpmodels_shim
DRY=${DRY:-0}
# abspath P : P as an absolute path (normalised if it exists; it need not exist yet)
abspath() { if [[ -d $1 ]]; then (cd "$1" && pwd); elif [[ $1 == /* ]]; then echo "$1"; else echo "$PWD/$1"; fi; }
OUT=$(abspath "${OUT:-$HERE/results/local_run}")   # created only once the checks below pass
GPU_TAG=${GPU_TAG:-h100}
EXTRAS=${EXTRAS:-1}
FORCE=${FORCE:-0}
ONLY=${ONLY:-"atomtrick mace gotennet equiformer equiformer_v2 mace_compiled gotennet_compiled equiformer_compiled equiformer_v2_compiled extras merge"}
COMPILE_VARIANTS=${COMPILE_VARIANTS:-"inductor fused"}
COMPILED_VARIANT=${COMPILED_VARIANT:-auto}
WARMUP=10; ITERS=50
RMD17_ALL="ethanol malonaldehyde benzene uracil toluene salicylic naphthalene paracetamol aspirin azobenzene"
MD22_ALL="Ac_Ala3_NHMe DHA AT_AT stachyose AT_AT_CG_CG buckyball_catcher double_walled_nanotube"
RMD17_MOLS=${RMD17_MOLS:-$RMD17_ALL}
MD22_MOLS=${MD22_MOLS:-$MD22_ALL}
nmols() { [[ $1 == rmd17 ]] && echo "$RMD17_MOLS" | wc -w || echo "$MD22_MOLS" | wc -w; }
# mols_arg DS : the harness --molecules flag for a subset (empty = all, i.e. the harness default)
mols_arg() {
  if [[ $1 == rmd17 ]]; then [[ "$RMD17_MOLS" == "$RMD17_ALL" ]] || echo "--molecules $RMD17_MOLS"
  else [[ "$MD22_MOLS" == "$MD22_ALL" ]] || echo "--molecules $MD22_MOLS"; fi
}

: "${RAW_ROOT:?set RAW_ROOT (directory with raw/rmd17/npz_data and raw/md22)}"
BD=${BASELINES_DIR:-}
EQUIFORMER_ROOT=${EQUIFORMER_ROOT:-${BD:+$BD/equiformer}}
EQUIFORMER_V2_ROOT=${EQUIFORMER_V2_ROOT:-${BD:+$BD/equiformer_v2}}
GOTENNET_ROOT=${GOTENNET_ROOT:-${BD:+$BD/GotenNet}}
MACE_ROOT=${MACE_ROOT:-${BD:+$BD/mace}}
EQUIFORMER_PREFIX=${EQUIFORMER_PREFIX-${EQUIFORMER_V2_PREFIX:-}}
EQUIFORMER_ACTIVATE=${EQUIFORMER_ACTIVATE-${EQUIFORMER_V2_ACTIVATE:-}}
MACE_COMPILED_PREFIX=${MACE_COMPILED_PREFIX-${MACE_PREFIX:-}}
MACE_COMPILED_ACTIVATE=${MACE_COMPILED_ACTIVATE-${MACE_ACTIVATE:-}}

want() { [[ " $ONLY " == *" $1 "* ]]; }
used=(RAW_ROOT)                                   # the directories the selected steps use
{ want mace || want mace_compiled; } && used+=(MACE_ROOT)
{ want gotennet || want gotennet_compiled; } && used+=(GOTENNET_ROOT)
{ want equiformer || want equiformer_compiled; } && used+=(EQUIFORMER_ROOT)
{ want equiformer_v2 || want equiformer_v2_compiled; } && used+=(EQUIFORMER_V2_ROOT)
miss=()
for v in "${used[@]}"; do [[ -z ${!v} ]] && miss+=("$v"); done
if [[ ${#miss[@]} -gt 0 ]]; then
  echo "ERROR: set BASELINES_DIR or $(printf '%s\n' "${miss[@]}" | sort -u | tr '\n' ' ')" >&2; exit 2
fi
# Absolute paths: every baseline command cd's into its clone before these are used.  A path
# that is not a directory is an error before anything runs (a warning under DRY=1).
bad=()
for v in RAW_ROOT EQUIFORMER_ROOT EQUIFORMER_V2_ROOT GOTENNET_ROOT MACE_ROOT; do
  [[ -n ${!v} ]] && printf -v "$v" '%s' "$(abspath "${!v}")"
done
for v in $(printf '%s\n' "${used[@]}" | sort -u); do [[ -d ${!v} ]] || bad+=("$v=${!v}"); done
if [[ ${#bad[@]} -gt 0 ]]; then
  if [[ $DRY != 1 ]]; then echo "ERROR: not a directory: ${bad[*]}" >&2; exit 2; fi
  echo "# WARNING: not a directory: ${bad[*]}"
fi

# the GPU every model times (see the header)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU_SEL=${CUDA_VISIBLE_DEVICES:-0}; GPU_SEL=${GPU_SEL%%,*}

# PyTorch defaults for every model: PYTORCH_JIT=0 would turn off TorchScript in e3nn/MACE
# (bench_atomtrick.py sets it for itself, and so does each baseline harness when run with
# --compile != none, before its import torch; the eager rows keep TorchScript on); an
# allocator config changes memory behaviour.
unset PYTORCH_JIT PYTORCH_CUDA_ALLOC_CONF
export PYTHONDONTWRITEBYTECODE=1                  # write nothing into the clones
if [[ $DRY != 1 ]]; then
  # fresh compiler caches, so the make_fx trace + Inductor compile (ATOMTRICK) and the Dynamo
  # trace + Inductor compile (compiled baselines) happen in the warmup calls
  JOB_CACHE=$(mktemp -d "${TMPDIR:-/tmp}/latency_run.XXXXXX")
  export TORCHINDUCTOR_CACHE_DIR=$JOB_CACHE/inductor TRITON_CACHE_DIR=$JOB_CACHE/triton
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  trap 'rm -rf "$JOB_CACHE"' EXIT
  mkdir -p "$OUT"/{raw,env,extra,logs}
  LOG=$OUT/driver.log
else
  LOG=/dev/null
fi
log() { echo "[latency $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
lf() { if [[ $DRY == 1 ]]; then echo /dev/null; else echo "$OUT/logs/$1.log"; fi; }  # step log
exec 3>&1                                         # DRY=1 prints commands here (not via a step log)
NFAIL=0                                           # step failures -> non-zero exit

# inenv M CMDSTRING : run CMDSTRING inside model M's environment (see M_PREFIX / M_ACTIVATE)
inenv() {
  local m=$1 cmd=$2 pv="${1}_PREFIX" av="${1}_ACTIVATE" pre act
  read -r -a pre <<< "${!pv:-}"
  act=${!av:-}
  # "ACT || exit 97; CMD": a failed activation ends the launch instead of running CMD (which
  # may hold several ';'-separated harness calls) in the outer environment
  if [[ $DRY == 1 ]]; then
    echo "+ ${pre[*]:+${pre[*]} }/bin/bash -lc \"${act:+$act || exit 97; }$cmd\"" >&3; return 0
  fi
  ${pre[@]+"${pre[@]}"} /bin/bash -lc "${act:+$act || exit 97; }$cmd"
}

# csv_rows FILE : "<data rows> <status counts>" of a harness CSV; the csv module handles quoted
# fields (a compile_error status may hold commas or newlines), wc/grep are the fallback
csv_rows() {
  python3 -c 'import csv, sys, collections
rs = list(csv.DictReader(open(sys.argv[1], newline="")))
c = collections.Counter((r.get("status") or "?").split(":")[0] for r in rs)
print(len(rs), " ".join(f"{v} {k}" for k, v in sorted(c.items())))' "$1" 2>/dev/null \
    || echo "$(( $(wc -l < "$1") - 1 )) $(tail -n +2 "$1" | grep -o 'ok\|oom\|error[^,]*' | sort | uniq -c | tr '\n' ' ')"
}
# finalize TMP FINAL NROWS : validate a harness CSV (header + NROWS data rows), move it into place.
# A compile_error / oom row is a row: a failed recipe does not fail the step.
finalize() {
  local tmp=$1 fin=$2 n=$3 got info
  [[ $DRY == 1 ]] && return 0
  if [[ ! -s "$tmp" ]]; then log "FAIL $fin: no output"; NFAIL=$((NFAIL + 1)); return 1; fi
  info=$(csv_rows "$tmp"); got=${info%% *}
  if [[ "$got" -ne "$n" ]]; then log "FAIL $fin: $got rows, expected $n (kept $tmp)"; NFAIL=$((NFAIL + 1)); return 1; fi
  mv -f "$tmp" "$fin"
  log "ok   $fin ($got rows; status: ${info#* })"
  # every row compile_error = the recipe never produced a number (setup failure or a recipe the
  # model cannot run); not a step failure, but worth a line of its own
  [[ "${info#* }" =~ ^[0-9]+\ compile_error\ *$ ]] && log "WARN $fin: every row is compile_error (see the step log)"
  return 0
}
need() { [[ "$FORCE" == 1 || ! -s "$1" ]]; }
rows_for() { echo $(( $(nmols "$1") * 2 )); }   # molecules x 2 batch sizes (20 / 14 for the full sets)

gpu_busy_check() {   # anything else on the timed GPU before a model is timed?
  [[ $DRY == 1 ]] && return 0
  local apps
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
         | { if [[ -n $GPU_UUID ]]; then grep -F "$GPU_UUID"; else cat; fi; })
  if [[ -n "$apps" ]]; then log "WARNING other compute processes on the GPU before $1: $apps"; fi
}

# ---------------------------------------------------------------- run metadata
if [[ $DRY != 1 ]]; then
  # GPU_SEL -> UUID; NSMI_I restricts nvidia-smi to it (falls back to the only GPU listed, else all)
  GPU_UUID=$(nvidia-smi -i "$GPU_SEL" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)
  if [[ $GPU_UUID != GPU-* ]]; then
    GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)
    [[ $(grep -c '^GPU-' <<< "$GPU_UUID") -eq 1 ]] || GPU_UUID=""
  fi
  if [[ -n $GPU_UUID ]]; then
    NSMI_I=(-i "$GPU_UUID")
  else
    NSMI_I=()
    log "WARNING cannot map CUDA_VISIBLE_DEVICES entry '$GPU_SEL' to one nvidia-smi GPU: querying all GPUs (conditions.py will then report more than one GPU)"
  fi
  {
    echo "gpu_selected=$GPU_SEL CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none} node=$(hostname)"
    echo "repo head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) dirty=$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
    for c in EQUIFORMER_ROOT EQUIFORMER_V2_ROOT GOTENNET_ROOT MACE_ROOT; do
      [[ -n ${!c} ]] && echo "clone_$c=$(git -C "${!c}" rev-parse HEAD 2>/dev/null) dirty_files=$(git -C "${!c}" status --porcelain --untracked-files=no 2>/dev/null | tr '\n' ' ')"
    done
    (cd "$HARNESS" && sha256sum latency_common.py */*.py) | sed 's/^/harness_sha256 /'
    sha256sum "$HERE/bench_atomtrick.py" | sed "s#$REPO/##; s/^/bench_atomtrick_sha256 /"
    echo "cpu=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p') cpus_allowed=$(nproc)"
    nvidia-smi ${NSMI_I[@]+"${NSMI_I[@]}"} --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,compute_mode,persistence_mode --format=csv
    echo "env: PYTORCH_JIT=${PYTORCH_JIT-unset} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
  } >> "$OUT/run_meta.txt" 2>&1
  nvidia-smi ${NSMI_I[@]+"${NSMI_I[@]}"} -q -d CLOCK,PERFORMANCE,POWER > "$OUT/logs/nvidia_smi_start.txt" 2>&1
  GPU_NAME=$(nvidia-smi ${NSMI_I[@]+"${NSMI_I[@]}"} --query-gpu=name --format=csv,noheader | head -1 | sed 's/^NVIDIA //')
  TITLE_GPU=${TITLE_GPU:-"$GPU_NAME, full fp32"}
  PLOT_GPU=${GPU_NAME%% *}
  log "start: $(hostname), GPU $(nvidia-smi ${NSMI_I[@]+"${NSMI_I[@]}"} --query-gpu=name,uuid --format=csv,noheader | head -1), OUT=$OUT"
else
  TITLE_GPU=${TITLE_GPU:-"<GPU name>, full fp32"}; PLOT_GPU="<GPU>"
  echo "# DRY RUN: commands only (OUT=$OUT)"
fi

PROBE="python $HERE/env_probe.py"
COMMON="--batch-sizes 1 4 --warmup $WARMUP --iters $ITERS --matmul-precision highest"

# ---------------------------------------------------------------- 1 ATOMTRICK
run_atomtrick() {   # $1 = output csv, rest = extra bench_atomtrick.py args
  local fin=$1; shift
  inenv ATOMTRICK "cd $REPO && python benchmarks/latency/bench_atomtrick.py --batch_sizes 1 4 --mode both \
--tf32 off --frame_source raw --raw_root $RAW_ROOT --warmup $WARMUP --iters $ITERS \
--config $REPO/configs/rmd17/atomtrick.yaml $(at_mols) $* --output $fin.tmp" \
    >> "$(lf "$(basename "$fin" .csv)")" 2>&1
  finalize "$fin.tmp" "$fin" $(( ($(nmols rmd17) + $(nmols md22)) * 4 ))   # molecules x 2 bs x 2 paths (68)
}
# at_mols : bench_atomtrick.py --molecules for a subset (it takes the names of both datasets at once)
at_mols() {
  if [[ "$RMD17_MOLS" != "$RMD17_ALL" || "$MD22_MOLS" != "$MD22_ALL" ]]; then echo "--molecules $RMD17_MOLS $MD22_MOLS"; fi
}
if want atomtrick && need "$OUT/atomtrick_latency.csv"; then
  gpu_busy_check atomtrick; log "ATOMTRICK (eager + compiled)"
  inenv ATOMTRICK "$PROBE atomtrick --repo-root $REPO --json $OUT/env/atomtrick.json > /dev/null \
&& pip freeze > $OUT/env/atomtrick_pip_freeze.txt" >> "$(lf env_probe)" 2>&1
  run_atomtrick "$OUT/atomtrick_latency.csv"
fi

# ---------------------------------------------------------------- 2-5 baselines
# baseline MODEL ENVKEY CLONE PRE SCRIPT ARGS... : both datasets in ONE environment launch
baseline() {
  local model=$1 key=$2 clone=$3 pre=$4 script=$5; shift 5
  local args="$*" todo=() ds
  local cmd="$pre $PROBE $model --repo-root $clone --json $OUT/env/$model.json > /dev/null && pip freeze > $OUT/env/${model}_pip_freeze.txt"
  for ds in rmd17 md22; do need "$OUT/raw/${model}_${ds}_latency_${GPU_TAG}.csv" && todo+=("$ds"); done
  [[ ${#todo[@]} -eq 0 ]] && { log "skip $model (done)"; return 0; }
  for ds in "${todo[@]}"; do
    cmd="$cmd; python $script --dataset $ds $args $(mols_arg "$ds") --output $OUT/raw/${model}_${ds}_latency_${GPU_TAG}.csv.tmp"
  done
  gpu_busy_check "$model"; log "$model: datasets ${todo[*]}"
  inenv "$key" "$cmd" >> "$(lf "$model")" 2>&1
  for ds in "${todo[@]}"; do
    finalize "$OUT/raw/${model}_${ds}_latency_${GPU_TAG}.csv.tmp" \
             "$OUT/raw/${model}_${ds}_latency_${GPU_TAG}.csv" "$(rows_for "$ds")"
  done
}

if want mace; then
  baseline mace MACE "$MACE_ROOT" "cd $MACE_ROOT &&" "$HARNESS/mace/benchmark_inference_mace.py" \
    --data-root "$RAW_ROOT" --dtype float32 $COMMON
fi
if want gotennet; then
  baseline gotennet GOTENNET "$GOTENNET_ROOT" "cd $GOTENNET_ROOT && export PROJECT_ROOT=$GOTENNET_ROOT &&" \
    "$HARNESS/gotennet/benchmark_inference.py" --repo-root "$GOTENNET_ROOT" --data-dir "$RAW_ROOT" \
    --create-graph false $COMMON
fi
if want equiformer; then
  baseline equiformer EQUIFORMER "$EQUIFORMER_ROOT" "cd $EQUIFORMER_ROOT && export PYTHONPATH=$SHIM &&" \
    "$HARNESS/equiformer/benchmark_inference.py" --repo-root "$EQUIFORMER_ROOT" \
    --coords raw --raw-root "$RAW_ROOT" --create-graph false $COMMON
fi
if want equiformer_v2; then
  baseline equiformer_v2 EQUIFORMER_V2 "$EQUIFORMER_V2_ROOT" "cd $EQUIFORMER_V2_ROOT && export PYTHONPATH=$SHIM &&" \
    "$HARNESS/equiformer_v2/benchmark_inference.py" --repo-root "$EQUIFORMER_V2_ROOT" \
    --coords raw --raw-root "$RAW_ROOT" --forces autograd $COMMON
fi

# ---------------------------------------------------------------- 6 compiled baselines
# baseline_compiled MODEL ENVKEY CLONE PRE SCRIPT ARGS... : the eager command plus --compile
# <variant>, one environment launch per variant (both datasets), outputs
# raw/<model>_compiled_<variant>_<ds>_latency_<GPU_TAG>.csv.  ENVKEY MACE_COMPILED also writes
# env/mace_compiled.json (+ pip freeze); the other models run in their eager environment.
baseline_compiled() {
  local model=$1 key=$2 clone=$3 pre=$4 script=$5; shift 5
  local args="$*" variant todo ds cmd stem probe=""
  [[ $key == MACE_COMPILED ]] && probe="$PROBE $model --repo-root $clone --json $OUT/env/${model}_compiled.json > /dev/null && pip freeze > $OUT/env/${model}_compiled_pip_freeze.txt; "
  for variant in $COMPILE_VARIANTS; do
    stem=$OUT/raw/${model}_compiled_${variant}
    todo=()
    for ds in rmd17 md22; do need "${stem}_${ds}_latency_${GPU_TAG}.csv" && todo+=("$ds"); done
    [[ ${#todo[@]} -eq 0 ]] && { log "skip ${model}_compiled ($variant done)"; continue; }
    cmd="$pre ${probe}true"
    for ds in "${todo[@]}"; do
      cmd="$cmd; python $script --dataset $ds $args $(mols_arg "$ds") --compile $variant --output ${stem}_${ds}_latency_${GPU_TAG}.csv.tmp"
    done
    gpu_busy_check "${model}_compiled"; log "${model}_compiled ($variant): datasets ${todo[*]}"
    inenv "$key" "$cmd" >> "$(lf "${model}_compiled_${variant}")" 2>&1
    for ds in "${todo[@]}"; do
      finalize "${stem}_${ds}_latency_${GPU_TAG}.csv.tmp" "${stem}_${ds}_latency_${GPU_TAG}.csv" "$(rows_for "$ds")"
    done
    probe=""
  done
}

if want mace_compiled; then
  baseline_compiled mace MACE_COMPILED "$MACE_ROOT" "cd $MACE_ROOT &&" "$HARNESS/mace/benchmark_inference_mace.py" \
    --data-root "$RAW_ROOT" --dtype float32 $COMMON
fi
if want gotennet_compiled; then
  baseline_compiled gotennet GOTENNET "$GOTENNET_ROOT" "cd $GOTENNET_ROOT && export PROJECT_ROOT=$GOTENNET_ROOT &&" \
    "$HARNESS/gotennet/benchmark_inference.py" --repo-root "$GOTENNET_ROOT" --data-dir "$RAW_ROOT" \
    --create-graph false $COMMON
fi
if want equiformer_compiled; then
  baseline_compiled equiformer EQUIFORMER "$EQUIFORMER_ROOT" "cd $EQUIFORMER_ROOT && export PYTHONPATH=$SHIM &&" \
    "$HARNESS/equiformer/benchmark_inference.py" --repo-root "$EQUIFORMER_ROOT" \
    --coords raw --raw-root "$RAW_ROOT" --create-graph false $COMMON
fi
if want equiformer_v2_compiled; then
  baseline_compiled equiformer_v2 EQUIFORMER_V2 "$EQUIFORMER_V2_ROOT" "cd $EQUIFORMER_V2_ROOT && export PYTHONPATH=$SHIM &&" \
    "$HARNESS/equiformer_v2/benchmark_inference.py" --repo-root "$EQUIFORMER_V2_ROOT" \
    --coords raw --raw-root "$RAW_ROOT" --forces autograd $COMMON
fi

# ---------------------------------------------------------------- 7 extra (not in the merged table)
if want extras && [[ "$EXTRAS" == 1 ]]; then
  if need "$OUT/extra/atomtrick_euler_latency.csv"; then
    gpu_busy_check atomtrick_euler; log "extra: ATOMTRICK with the Euler edge frame"
    run_atomtrick "$OUT/extra/atomtrick_euler_latency.csv" --config "$HERE/escn_euler.yaml"
  fi
fi

# ---------------------------------------------------------------- 8 merge / plot / conditions
[[ $DRY != 1 ]] && nvidia-smi ${NSMI_I[@]+"${NSMI_I[@]}"} -q -d CLOCK,PERFORMANCE,POWER > "$OUT/logs/nvidia_smi_end.txt" 2>&1
if want merge; then
  log "merge + plot + conditions"
  inenv ATOMTRICK "cd $REPO && python benchmarks/latency/merge.py --atomtrick_csv $OUT/atomtrick_latency.csv \
--baseline_dir $OUT/raw --gpu_tag $GPU_TAG --compiled_variant $COMPILED_VARIANT --title_gpu '$TITLE_GPU' \
--atomtrick_note 'strict fp32 (matmul precision highest); eager = autograd forces, create_graph=False; compiled = make_fx trace (train_mode) + torch.compile(dynamic=True), run under no_grad' \
--out_long $OUT/latency_all_long.csv --out_md $OUT/latency_all_summary.md && \
python benchmarks/latency/plot.py --long $OUT/latency_all_long.csv --out $OUT/latency_all \
--title '$PLOT_GPU inference latency (energy + forces), full fp32: ATOMTRICK vs baselines'" \
    >> "$(lf merge)" 2>&1 \
    && log "ok   merge: $OUT/latency_all_summary.md, $OUT/latency_all.{png,pdf}" \
    || { log "FAIL merge/plot (see $OUT/logs/merge.log)"; NFAIL=$((NFAIL + 1)); }
  if [[ $DRY == 1 ]]; then
    echo "+ python3 $HERE/conditions.py $OUT --gpu_tag $GPU_TAG"
  else
    python3 "$HERE/conditions.py" "$OUT" --gpu_tag "$GPU_TAG" >> "$OUT/logs/conditions.log" 2>&1 \
      && log "ok   conditions: $OUT/MEASUREMENT_CONDITIONS.md, $OUT/latency_all_summary_d5.md" \
      || { log "CHECK conditions reported problems: see $OUT/MEASUREMENT_CONDITIONS.md"; NFAIL=$((NFAIL + 1)); }
  fi
fi
log "done; failures: $NFAIL"
exit $(( NFAIL > 0 ? 1 : 0 ))
