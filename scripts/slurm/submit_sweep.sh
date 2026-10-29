#!/bin/bash
# Submit the hyperparameter sweep for one model: one job per (dataset, readout), each
# running scripts/slurm/run_sweep.slurm. Jobs resume where they stopped, so resubmitting
# after a time-out or preemption is safe (finished runs are skipped).
#
#   bash scripts/slurm/submit_sweep.sh chemprop                 # all its datasets x readouts
#   bash scripts/slurm/submit_sweep.sh gine BBBP ESOL           # a subset of datasets
#   bash scripts/slurm/submit_sweep.sh chemprop --dry-run       # print the sbatch commands only
#   SWEEP_READOUTS="mean sum" bash scripts/slurm/submit_sweep.sh gine   # a subset of readouts
#
# Split files for seeds 0-4 must already exist in data/splits/ (see the protocol).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

MODEL=$1
shift
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; shift; fi
DATASETS=("$@")

# Readouts and datasets per model, from docs/hparam_sweep_protocol.ipynb
case "$MODEL" in
    gin|gine) READOUTS=(max mean sum attention multihead)
              ALL_DATASETS=(BBBP BACE CLINTOX) ;;
    chemprop) READOUTS=(max mean sum attention)
              ALL_DATASETS=(BBBP BACE CLINTOX) ;;   # Chemprop baseline is binary-only
    *)        echo "Unknown model '$MODEL'. Choose from: gin, gine, chemprop" >&2; exit 1 ;;
esac
if [[ ${#DATASETS[@]} -eq 0 ]]; then DATASETS=("${ALL_DATASETS[@]}"); fi
if [[ -n "${SWEEP_READOUTS:-}" ]]; then read -ra READOUTS <<< "$SWEEP_READOUTS"; fi

for DS in "${DATASETS[@]}"; do
    for RO in "${READOUTS[@]}"; do
        CMD=(sbatch --parsable --job-name="sweep_${MODEL}_${DS,,}_${RO}"
             scripts/slurm/run_sweep.slurm "$DS" "$MODEL" "$RO")
        if $DRY_RUN; then echo "${CMD[*]}"; continue; fi
        ID=$("${CMD[@]}")
        echo "submitted $DS/$MODEL/$RO -> job $ID"
    done
done
