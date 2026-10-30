#!/usr/bin/env bash
# Evaluate every trained run under runs/ on its test set and build the result tables.
#
# For each directory with an args.json (written by train.py) the test set is derived from the
# run's training file and the run's *_best.pt (EMA weights at the best validation loss) is scored:
#   rMD17   data/rmd17/<mol>_test_default.npz      (1000 structures)   --eval_bs 8
#   MD22    data/md22/<system>_test_default.npz    (all held-out frames) --eval_bs 8/8/8/4/4/2/1
#   SPICE   data/spice_maceoff/spice_maceoff_test_f15.npz (force-filtered, 50,069 structures)
#           --per_subset --eval_bs 16
# writing <run_dir>/eval_test.json (SPICE: eval_test_f15.json).  Existing JSONs are
# kept unless --force.  Then the aggregation scripts turn the JSONs into markdown tables
# (mean +- sample std over seeds, comparison with published baselines):
#   scripts/aggregate/rmd17_table.py, md22_table.py, spice_table.py  -> results/*.md
#
# Usage:
#   bash scripts/reproduce/eval_all.sh [--dry] [--force] [--runs DIR] [--only SUBSTRING]
#                                      [--no-aggregate | --aggregate-only]
#   --runs DIR      directory to scan (default: runs)
#   --only S        only runs whose path contains S (e.g. --only rmd17_so2)
#
# Evaluation labels: evaluate.py computes labels in float64, as the reported numbers do.
# Matmul precision: each run is scored with the precision recorded in its args.json (default:
# full fp32); an args.json that records none is scored with TF32.
#
# Cost: minutes per run on one GPU (the largest test sets, Ac-Ala3-NHMe with 78,609 frames and
# SPICE with 50,069, take ~10-15 min each on an L40S).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set -euo pipefail

RUNS_DIR=runs
ONLY=""
AGGREGATE=1
EVALUATE=1
RESULTS_DIR=${RESULTS_DIR:-results}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage_from_header; exit 0 ;;
        --runs) [[ $# -ge 2 ]] || die "--runs needs a value"; RUNS_DIR=$2; shift 2 ;;
        --only) [[ $# -ge 2 ]] || die "--only needs a value"; ONLY=$2; shift 2 ;;
        --no-aggregate) AGGREGATE=0; shift ;;
        --aggregate-only) EVALUATE=0; shift ;;
        *) parse_common_opt "$@"; shift "$SHIFT_N" ;;
    esac
done

# ---------------------------------------------------------------------------- evaluation
if [[ $EVALUATE == 1 ]]; then
    list_args=(--runs "$RUNS_DIR")
    if [[ -n "$ONLY" ]]; then list_args+=(--only "$ONLY"); fi
    runs_tsv=$("$PYTHON" scripts/reproduce/_runs.py "${list_args[@]}")
    if [[ -z "$runs_tsv" ]]; then
        log "no evaluable runs under $RUNS_DIR/"
    fi
    # read from fd 3 so evaluate.py never inherits the run list on stdin
    while IFS=$'\t' read -r -u 3 ckpt data mol bs out tag per_subset; do
        # ignore anything that is not a complete 7-field record (e.g. interpreter start-up noise)
        [[ -z "${per_subset:-}" || -z "${ckpt:-}" ]] && continue
        if [[ "$mol" == "-" ]]; then mol=""; fi
        extra=()
        if [[ "$per_subset" == 1 ]]; then extra+=(--per_subset); fi
        eval_unit "$ckpt" "$data" "$mol" "$bs" "$out" "$tag" ${extra[@]+"${extra[@]}"}
    done 3<<< "$runs_tsv"
fi

# ---------------------------------------------------------------------------- aggregation
# Each aggregator receives the evaluate.py JSON files of its benchmark as positional arguments and
# prints a markdown table, saved to $RESULTS_DIR/<benchmark>.md.
aggregate_one() {   # aggregate_one SCRIPT OUT_MD JSON...
    local script=$1 out=$2
    shift 2
    if [[ ! -f "$script" ]]; then
        log "[warn] $script not found; skipping"
        return 0
    fi
    if [[ $# -eq 0 ]]; then
        log "[skip] no results for $(basename "$script")"
        return 0
    fi
    echo "+ $PYTHON $script <$# result files> > $out"
    if [[ $DRY == 1 ]]; then return 0; fi
    mkdir -p "$(dirname "$out")"
    "$PYTHON" "$script" "$@" | tee "$out"
}

if [[ $AGGREGATE == 1 ]]; then
    shopt -s nullglob
    rmd17_json=("$RUNS_DIR"/rmd17_*/*/s*/eval_test.json)
    md22_json=("$RUNS_DIR"/md22_*/*/s*/eval_test.json)
    spice_json=("$RUNS_DIR"/spice_*/s*/eval_test_f15.json)
    shopt -u nullglob
    aggregate_one scripts/aggregate/rmd17_table.py "$RESULTS_DIR/rmd17.md" ${rmd17_json[@]+"${rmd17_json[@]}"}
    aggregate_one scripts/aggregate/md22_table.py "$RESULTS_DIR/md22.md" ${md22_json[@]+"${md22_json[@]}"}
    aggregate_one scripts/aggregate/spice_table.py "$RESULTS_DIR/spice.md" ${spice_json[@]+"${spice_json[@]}"}
fi
finish
