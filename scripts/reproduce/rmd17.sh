#!/usr/bin/env bash
# Reproduce the rMD17 results of the ATOMTRICK paper model (the paper's rMD17 energy/force MAE
# and force-L2 tables): 10 molecules x seeds 42, 1, 2 with
# configs/rmd17/atomtrick.yaml, then score every run's best (EMA) checkpoint on the 1000-structure
# official test split.
#
# Usage:
#   bash scripts/reproduce/rmd17.sh [--dry] [--train-only | --eval-only] [--force]
#                                   [--seeds "42 1 2"] [--molecules "aspirin ethanol ..."]
#
#   --dry         print the commands without running them
#   --train-only / --eval-only   skip the other phase
#   --force       retrain / re-evaluate even if outputs exist (default: finished runs are
#                 skipped and interrupted runs resume from their rolling checkpoint)
#
# Outputs: runs/rmd17_atomtrick/<molecule>/s<seed>/{args.json, *_best.pt, *_final.pt, eval_test.json}
# Then:    bash scripts/reproduce/eval_all.sh     (aggregates every evaluated run into tables)
#
# Cost of the reported runs: 9.1-12.6 GPU-hours per molecule and seed on one NVIDIA L40S
# (median 9.5 h; aspirin seed 42 9 h 05 min), i.e. ~305 L40S-hours for all 30 runs; evaluation takes
# minutes.  The 30 runs execute one after another on the visible GPU; split the work over
# several jobs with --molecules / --seeds.
#
# Needs: data/rmd17/<molecule>_{train,val,test}_default.npz (scripts/data/prepare_rmd17.py).
# Env:   PYTHON (default python), EVAL_ARGS (see scripts/reproduce/_common.sh)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set -euo pipefail

CONFIG=configs/rmd17/atomtrick.yaml

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage_from_header; exit 0 ;;
        *) parse_common_opt "$@"; shift "$SHIFT_N" ;;
    esac
done
SEEDS=${SEEDS:-"42 1 2"}
MOLECULES=${MOLECULES:-$RMD17_MOLECULES}

molecular_pipeline data/rmd17 "$CONFIG"
finish
