#!/usr/bin/env bash
# Reproduce the SPICE (MACE-OFF23 split) model of the comparison with MACE-OFF23 S/M/L:
#   stage 1  configs/spice/stage1_force.yaml   force-weighted training to 900k steps, run as a
#            resume chain of segments (SPICE_CHAIN, default "300000 600000 900000", so that each
#            job fits a cluster wall-time limit);
#   stage 2  + configs/spice/stage2_swa.yaml   resume at 900k, energy-weighted SWA stage to 920k;
#   eval     stage 2's *_best.pt on the force-filtered test set spice_maceoff_test_f15.npz
#            (MACE nonphysical-geometry filter, 50,069 structures; the only SPICE test set evaluated
#            and reported), per subset, with
#            evaluate.py --per_subset --eval_bs 16.
#
# Usage:
#   bash scripts/reproduce/spice.sh [--dry] [--train-only | --eval-only] [--force]
#                                   [--seeds 42] [--steps "stage1 stage2 eval"] [--segment N]
#                                   [--chain "300000 600000 900000"]
#
#   --steps      subset of stage1 / stage2 / eval to run (default: all; --train-only = stage1
#                stage2, --eval-only = eval)
#   --segment N  run only the stage-1 segment that ends at step N (to run the segments as
#                separate jobs; pass the same --chain to every job)
#   --chain "N1 N2 ..."  stage-1 segment ends (overrides SPICE_CHAIN)
#   other options as in scripts/reproduce/rmd17.sh
#   SPICE_CHAIN="900000" (or --chain 900000) trains stage 1 in one go (equivalent up to data order;
#   ~100 h on an H200).
#
# Completed stage-1 segments are recorded as runs/spice_stage1_force/s<seed>/.stage1_done_<N>;
# an interrupted segment resumes from the newer of the rolling and the final checkpoint.
#
# Outputs: runs/spice_stage1_force/s42/ (stage 1), runs/spice_stage2_swa/s42/ (reported model;
#          eval_test_f15.json); then scripts/reproduce/eval_all.sh aggregates.
#
# Cost of the reported run: one NVIDIA H200 141 GB, 16 CPUs, 200 GB host RAM; ~33 h per
# 300k steps + 2.4 h stage 2 = ~102 H200-hours (stage 1 trained as one 900k schedule; 99.9 h
# including two restarts); evaluation 13 min on an L40S.
#
# Needs: data/spice_maceoff/spice_maceoff_{train,val,test_f15}.npz
#        (scripts/data/prepare_spice_maceoff.py).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set -euo pipefail

STAGE1=configs/spice/stage1_force.yaml
STAGE2=configs/spice/stage2_swa.yaml
SPICE_CHAIN=${SPICE_CHAIN:-"300000 600000 900000"}
STEPS=""
SEGMENT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage_from_header; exit 0 ;;
        --steps)   [[ $# -ge 2 ]] || die "--steps needs a value"; STEPS=$2; shift 2 ;;
        --segment) [[ $# -ge 2 ]] || die "--segment needs a value"; SEGMENT=$2; shift 2 ;;
        --chain)   [[ $# -ge 2 ]] || die "--chain needs a value"; SPICE_CHAIN=$2; shift 2 ;;
        *) parse_common_opt "$@"; shift "$SHIFT_N" ;;
    esac
done
SEEDS=${SEEDS:-42}
if [[ -z "$STEPS" ]]; then
    STEPS=""
    if [[ $DO_TRAIN == 1 ]]; then STEPS="stage1 stage2"; fi
    if [[ $DO_EVAL == 1 ]]; then STEPS="$STEPS eval"; fi
fi
has_step() { [[ " $STEPS " == *" $1 "* ]]; }
[[ -n "$MOLECULES" ]] && log "note: --molecules is ignored for SPICE"

# Normalise the chain to single-space-separated step counts (its last word is the step stage 2
# resumes from).
read -r -a chain_words <<< "$SPICE_CHAIN"
[[ ${#chain_words[@]} -gt 0 ]] || die "empty stage-1 chain (SPICE_CHAIN / --chain)"
for seg in "${chain_words[@]}"; do
    [[ "$seg" =~ ^[0-9]+$ ]] || die "stage-1 chain entry '$seg' is not a step count"
done
SPICE_CHAIN="${chain_words[*]}"
if [[ -n "$SEGMENT" && " $SPICE_CHAIN " != *" $SEGMENT "* ]]; then
    die "--segment $SEGMENT is not in the stage-1 chain \"$SPICE_CHAIN\""
fi

EXP1=$(config_experiment "$STAGE1")
EXP2=$(config_experiment "$STAGE1" "$STAGE2")
LAST_SEG=${SPICE_CHAIN##* }

for seed in $SEEDS; do
    run1="runs/$EXP1/s$seed"; name1="${EXP1}_s${seed}"
    run2="runs/$EXP2/s$seed"; name2="${EXP2}_s${seed}"

    if has_step stage1; then
        first=1
        for seg in $SPICE_CHAIN; do
            if [[ -n "$SEGMENT" && "$seg" != "$SEGMENT" ]]; then
                first=0
                continue
            fi
            marker="$run1/.stage1_done_$seg"
            if [[ -f "$marker" && $FORCE == 0 ]]; then
                log "[skip] stage-1 segment $seg done ($marker)"
                first=0
                continue
            fi
            cmd=("$PYTHON" train.py --config "$STAGE1" --seed "$seed" --max_iterations "$seg")
            final="$run1/${name1}_final.pt"
            rolling="$run1/${name1}.pt"
            if [[ $first == 1 ]]; then
                if [[ -f "$rolling" && $FORCE == 0 ]]; then
                    log "[resume] interrupted first segment: $rolling"
                    cmd+=(--resume "$rolling")
                fi
            else
                if [[ -f "$rolling" && -f "$final" && "$rolling" -nt "$final" && $FORCE == 0 ]]; then
                    log "[resume] interrupted segment $seg: $rolling"
                    cmd+=(--resume "$rolling")
                elif [[ -f "$final" || $DRY == 1 ]]; then
                    cmd+=(--resume "$final")
                else
                    die "segment $seg needs the previous segment's $final"
                fi
            fi
            run "${cmd[@]}"
            if [[ $DRY == 0 ]]; then touch "$marker"; fi
            first=0
        done
    fi

    if has_step stage2; then
        if [[ $DRY == 0 && ! -f "$run1/.stage1_done_$LAST_SEG" ]]; then
            die "stage 1 has not reached step $LAST_SEG ($run1/.stage1_done_$LAST_SEG missing)"
        fi
        train_unit "$run2" "$name2" "$PYTHON" train.py --config "$STAGE1" --config "$STAGE2" --seed "$seed"
    fi

    if has_step eval; then
        eval_unit "$run2/${name2}_best.pt" "data/spice_maceoff/spice_maceoff_test_f15.npz" "" 16 \
                  "$run2/eval_test_f15.json" "$EXP2" --per_subset
    fi
done
finish
