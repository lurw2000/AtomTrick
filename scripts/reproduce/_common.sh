# Shared helpers for scripts/reproduce/*.sh.  Source this file; do not run it.
#
# Every reproduce script runs its commands here, one after another, on the visible GPU;
# --dry prints them instead.  Finished runs are skipped and interrupted runs resume from their
# rolling checkpoint, so a script can be re-run after a timeout; split the work over several
# jobs with --molecules / --seeds (SPICE: --segment).
#
# Environment:
#   PYTHON          interpreter for train.py / evaluate.py (default: python)
#   EVAL_ARGS       extra evaluate.py flags, e.g. EVAL_ARGS="--eval_bs 4".  Matmul precision needs
#                   no flag: evaluate.py uses the precision recorded in each run's args.json
#                   (default: full fp32, 'highest'; TF32, 'high', for an args.json that records
#                   none); EVAL_ARGS="--matmul_precision ..." overrides it for every run.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
DRY=0
DO_TRAIN=1
DO_EVAL=1
FORCE=0
SEEDS=""
MOLECULES=""
SHIFT_N=1
N_MISSING=0
N_NO_DATA=0

RMD17_MOLECULES="aspirin azobenzene benzene ethanol malonaldehyde naphthalene paracetamol salicylic toluene uracil"
MD22_MOLECULES="Ac-Ala3-NHMe DHA AT-AT stachyose AT-AT-CG-CG buckyball-catcher double-walled_nanotube"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(basename "$0")] $*" >&2; }

# parse_common_opt ARG [NEXT] -- handle one of the shared options; sets SHIFT_N (1 or 2).
parse_common_opt() {
    SHIFT_N=1
    case "$1" in
        --dry)        DRY=1 ;;
        --train-only) DO_EVAL=0 ;;
        --eval-only)  DO_TRAIN=0 ;;
        --force)      FORCE=1 ;;
        --seeds)      [[ $# -ge 2 ]] || die "--seeds needs a value"; SEEDS=$2; SHIFT_N=2 ;;
        --molecules)  [[ $# -ge 2 ]] || die "--molecules needs a value"; MOLECULES=$2; SHIFT_N=2 ;;
        *)            die "unknown option '$1' (see --help)" ;;
    esac
}

# quote_cmd ARG...  -- the command as a copy-pasteable shell line (single quotes only where needed).
quote_cmd() {
    local out="" a
    for a in "$@"; do
        if [[ "$a" =~ ^[A-Za-z0-9_./:=,+@%-]+$ ]]; then
            out+="$a "
        else
            out+="'${a//\'/\'\\\'\'}' "
        fi
    done
    echo "${out% }"
}

# run CMD...  -- echo the command; execute it unless --dry.
run() {
    echo "+ $(quote_cmd "$@")"
    if [[ $DRY == 1 ]]; then
        return 0
    fi
    "$@"
}

# config_experiment FILE...  -- the `experiment:` value of the last file that sets one.
config_experiment() {
    local f exp="" v
    for f in "$@"; do
        [[ -f "$f" ]] || die "config not found: $f"
        v=$(sed -n 's/^experiment:[[:space:]]*//p' "$f" | tail -n 1 | sed 's/[[:space:]]*#.*$//; s/^"//; s/"$//')
        if [[ -n "$v" ]]; then exp=$v; fi
    done
    [[ -n "$exp" ]] || die "no 'experiment:' key in $*"
    echo "$exp"
}

# train_unit RUN_DIR NAME CMD...  -- train unless <RUN_DIR>/<NAME>_final.pt exists; resume from the
# rolling checkpoint <NAME>.pt if a previous attempt was interrupted.
train_unit() {
    local run_dir=$1 name=$2
    shift 2
    local cmd=("$@")
    if [[ -f "$run_dir/${name}_final.pt" && $FORCE == 0 ]]; then
        log "[skip] $run_dir/${name}_final.pt exists (use --force to retrain)"
        return 0
    fi
    if [[ -f "$run_dir/${name}.pt" && $FORCE == 0 ]]; then
        log "[resume] found interrupted run $run_dir/${name}.pt"
        cmd+=(--resume "$run_dir/${name}.pt")
    fi
    run "${cmd[@]}"
}

# eval_unit CKPT DATA MOL EVAL_BS OUT TAG [evaluate.py flags...]
#   CKPT / DATA may contain {molecule}; MOL (may be empty) is then passed as --molecules.
#   Skips when OUT exists (unless --force) or the checkpoint is missing (counted in N_MISSING);
#   a missing test file is counted in N_NO_DATA and makes finish() exit with an error.
eval_unit() {
    local ckpt=$1 data=$2 mol=$3 bs=$4 out=$5 tag=$6
    shift 6
    local ckpt_file=${ckpt//\{molecule\}/$mol}
    local data_file=${data//\{molecule\}/$mol}
    if [[ -f "$out" && $FORCE == 0 ]]; then
        log "[skip] $out exists (use --force to re-evaluate)"
        return 0
    fi
    if [[ $DRY == 0 && ! -f "$ckpt_file" ]]; then
        log "[warn] checkpoint not found, skipping: $ckpt_file"
        N_MISSING=$((N_MISSING + 1))
        return 0
    fi
    if [[ $DRY == 0 && ! -f "$data_file" ]]; then
        log "[error] test data not found: $data_file (run scripts/data/prepare_*.py first); skipping"
        N_NO_DATA=$((N_NO_DATA + 1))
        return 0
    fi
    local cmd=("$PYTHON" evaluate.py --checkpoint "$ckpt" --data "$data" --eval_bs "$bs"
               --out "$out" --tag "$tag")
    if [[ -n "$mol" ]]; then cmd+=(--molecules "$mol"); fi
    cmd+=("$@")
    if [[ -n "${EVAL_ARGS:-}" ]]; then
        local extra
        read -r -a extra <<< "$EVAL_ARGS"
        cmd+=("${extra[@]}")
    fi
    run "${cmd[@]}"
}

# md22_eval_bs SYSTEM  -- evaluation batch size that fits in GPU memory (as used for the paper).
md22_eval_bs() {
    case "$1" in
        Ac-Ala3-NHMe|DHA|AT-AT)       echo 8 ;;
        stachyose|AT-AT-CG-CG)        echo 4 ;;
        buckyball-catcher)            echo 2 ;;
        double-walled_nanotube)       echo 1 ;;
        *)                            echo 4 ;;
    esac
}

# ---------------------------------------------------------------------------------------------
# Per-molecule pipeline (rMD17 / MD22 style): train every (seed, molecule), then evaluate each run's
# *_best.pt on <data_dir>/<molecule>_test_default.npz, writing <run_dir>/eval_test.json.
#
# Hook a script may redefine after sourcing this file:
#   eval_bs_for MOL      evaluate.py --eval_bs                       (default: 8)
eval_bs_for() { echo 8; }

# molecular_pipeline DATA_DIR CONFIG...
#   Uses the globals SEEDS and MOLECULES (space-separated lists).
molecular_pipeline() {
    local data_dir=$1
    shift 1
    local configs=("$@")
    local exp cfg_args=() c seed mol run name
    exp=$(config_experiment "${configs[@]}")
    for c in "${configs[@]}"; do cfg_args+=(--config "$c"); done

    for seed in $SEEDS; do
        for mol in $MOLECULES; do
            run="runs/$exp/$mol/s$seed"
            name="${exp}_${mol}_s${seed}"
            if [[ $DO_TRAIN == 1 ]]; then
                train_unit "$run" "$name" "$PYTHON" train.py "${cfg_args[@]}" --molecule "$mol" --seed "$seed"
            fi
            if [[ $DO_EVAL == 1 ]]; then
                eval_unit "runs/$exp/{molecule}/s$seed/${exp}_{molecule}_s${seed}_best.pt" \
                          "$data_dir/{molecule}_test_default.npz" "$mol" "$(eval_bs_for "$mol")" \
                          "$run/eval_test.json" "$exp"
            fi
        done
    done
}

# usage_from_header  -- print the calling script's leading comment block.
usage_from_header() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
}

finish() {
    if [[ $N_MISSING -gt 0 ]]; then
        log "$N_MISSING evaluation(s) skipped because the checkpoint does not exist yet"
    fi
    if [[ $DRY == 1 ]]; then
        log "dry run: nothing was executed"
    fi
    if [[ $N_NO_DATA -gt 0 ]]; then
        die "$N_NO_DATA evaluation(s) failed because the test data file is missing"
    fi
}
