#!/usr/bin/env bash
# Reproduce the MD22 results of the ATOMTRICK paper model (the paper's MD22 energy/force MAE
# and force-L2 tables): the 7 systems with configs/md22/atomtrick.yaml
# (AdamW, one configuration for all systems).  Every run is then scored on its test split (all
# frames not used for training/validation) with the memory-limited evaluation batch sizes
# 8/8/8/4/4/2/1 (Ac-Ala3-NHMe, DHA, AT-AT, stachyose, AT-AT-CG-CG, buckyball-catcher, nanotube).
#
# Usage:
#   bash scripts/reproduce/md22.sh [--dry] [--train-only | --eval-only] [--force]
#                                  [--variants adamw] [--seeds "42"] [--molecules "DHA ..."]
#
#   --variants   adamw = configs/md22/atomtrick.yaml for every selected system (the only
#                variant; the default).
#   other options as in scripts/reproduce/rmd17.sh (default seed 42)
#
# Outputs: runs/md22_atomtrick/<system>/s42/..., each with eval_test.json; then
#          scripts/reproduce/eval_all.sh aggregates.
#
# Cost of the reported runs (one GPU per system, make_fx, batch size 4, 750k steps):
#   H100 80 GB : Ac-Ala3-NHMe 10.6 h, DHA 11.2 h, AT-AT 11.1 h
#   H200 141 GB: stachyose 16.7 h, AT-AT-CG-CG 16.4 h, buckyball-catcher 25.3 h,
#                double-walled_nanotube 63.7 h
#   (the stachyose, AT-AT-CG-CG and buckyball-catcher times include one restart from the rolling
#   checkpoint).  Total ~155 GPU-hours.  The four large systems need a large-memory GPU (the
#   nanotube needs an H200-class card at batch size 4 on the make_fx path).  The nanotube run
#   exceeds 48 h: re-run the script after a timeout, interrupted runs resume from their rolling
#   checkpoint.
#
# Needs: data/md22/<system>_{train,val,test}_default.npz (scripts/data/prepare_md22.py).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set -euo pipefail

VARIANTS="adamw"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage_from_header; exit 0 ;;
        --variants) [[ $# -ge 2 ]] || die "--variants needs a value"; VARIANTS=$2; shift 2 ;;
        *) parse_common_opt "$@"; shift "$SHIFT_N" ;;
    esac
done
SEEDS=${SEEDS:-42}
MOLECULES=${MOLECULES:-$MD22_MOLECULES}
ALL_SELECTED=$MOLECULES

eval_bs_for() { md22_eval_bs "$1"; }
for variant in $VARIANTS; do
    [[ "$variant" == adamw ]] || die "unknown variant '$variant' (the only variant is adamw)"
done
MOLECULES=$ALL_SELECTED
molecular_pipeline data/md22 configs/md22/atomtrick.yaml
finish
