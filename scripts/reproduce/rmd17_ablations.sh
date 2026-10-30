#!/usr/bin/env bash
# Reproduce the rMD17 ablations of the ATOMTRICK paper model: each NAME trains
# configs/rmd17/atomtrick.yaml + configs/rmd17/ablations/NAME.yaml on the 10 molecules and scores
# every run on the official test split.  Compare against scripts/reproduce/rmd17.sh (the O(2) model).
#
# Usage:
#   bash scripts/reproduce/rmd17_ablations.sh [options] NAME [NAME ...]     # or: all
#   options: --dry --train-only --eval-only --force --seeds "..." --molecules "..."
#            (same meaning as in scripts/reproduce/rmd17.sh)
#
# NAME (config header = parameter count, hardware and reported result):
#   so2           SO(2)- instead of O(2)-linear K/V   paper's main ablation          default seeds 42 1 2
#   softmax       softmax attention weights           ablation table                 default seed 42
#   perdegree_ln  per-degree LayerNorm                ablation table                 default seed 42
#   o3_kv lmax1 lmax2 layers4 layers8 layers12
#                 the remaining ablation-table rows                                   seed 42
#   --seeds overrides the default seeds for every NAME.
#
# Outputs: runs/rmd17_<NAME>/<molecule>/s<seed>/... + eval_test.json; then eval_all.sh aggregates.
#
# Cost of the reported runs (one NVIDIA L40S), median per molecule and seed: so2 10.9 h
# (larger make_fx graph), softmax / perdegree_ln 9.5 h, o3_kv 6.8 h, lmax1 6.1 h,
# lmax2 7.7 h, layers4 / layers8 / layers12 6.7 / 11.7 / 17.3 h; the slowest molecule took up to
# ~2.5 h (so2: 4.7 h) longer.  so2 with 3 seeds ~ 340 L40S-hours; a single-seed ablation 60-175
# L40S-hours.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set -euo pipefail

ALL_NAMES="so2 softmax perdegree_ln o3_kv lmax1 lmax2 layers4 layers8 layers12"

NAMES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage_from_header; exit 0 ;;
        all) read -r -a NAMES <<< "$ALL_NAMES"; shift ;;
        --*) parse_common_opt "$@"; shift "$SHIFT_N" ;;
        *) NAMES+=("$1"); shift ;;
    esac
done
if [[ ${#NAMES[@]} -eq 0 ]]; then
    usage_from_header
    die "give at least one ablation NAME (or 'all')"
fi
MOLECULES=${MOLECULES:-$RMD17_MOLECULES}
USER_SEEDS=$SEEDS

for NAME in "${NAMES[@]}"; do
    override="configs/rmd17/ablations/$NAME.yaml"
    [[ -f "$override" ]] || die "unknown ablation '$NAME' (no $override); known: $ALL_NAMES"
    if [[ -n "$USER_SEEDS" ]]; then
        SEEDS=$USER_SEEDS
    elif [[ "$NAME" == so2 ]]; then
        SEEDS="42 1 2"
    else
        SEEDS="42"
    fi
    log "ablation $NAME: seeds $SEEDS"
    molecular_pipeline data/rmd17 configs/rmd17/atomtrick.yaml "$override"
done
finish
