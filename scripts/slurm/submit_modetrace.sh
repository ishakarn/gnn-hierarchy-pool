#!/bin/bash
# Submit ModeTrace runs for GIN/GINE, one job per dataset (OOD) and per
# dataset x model (matched), into results/modetrace/. Jobs resume where they
# stopped, so resubmitting after a preemption is safe.
#
#   bash scripts/slurm/submit_modetrace.sh                       # all datasets
#   bash scripts/slurm/submit_modetrace.sh ESOL LIPO             # a subset
#   bash scripts/slurm/submit_modetrace.sh --dry-run
#
# Override the matched sweep with env vars, e.g. the full protocol:
#   MATCHED_P="0.25 0.5 0.75 1.0" MATCHED_FOLDS="0 1 2 3 4 5 6 7 8 9" bash scripts/slurm/submit_modetrace.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; shift; fi
DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(BBBP BACE CLINTOX ESOL FREESOLV LIPO)

MODELS=(${MODELS:-gin gine})
READOUTS=${READOUTS:-mean}
PERTS="WNFS GNFS WEFS GEFS JGFS"
MATCHED_P=${MATCHED_P:-"0.5 1.0"}
MATCHED_FOLDS=${MATCHED_FOLDS:-"0 1 2"}

submit() {   # name time args...
    local name=$1 time=$2; shift 2
    local cmd=(sbatch --parsable --job-name="$name" --time="$time" scripts/slurm/run_modetrace.slurm "$@")
    if $DRY_RUN; then echo "${cmd[*]}"; else echo "$name -> job $("${cmd[@]}")"; fi
}

for DS in "${DATASETS[@]}"; do
    submit "mt_ood_${DS,,}" 4:00:00 ood \
        --datasets "$DS" --models "${MODELS[@]}" --readouts $READOUTS --perturbations $PERTS
    for M in "${MODELS[@]}"; do
        submit "mt_matched_${DS,,}_${M}" 8:00:00 matched \
            --datasets "$DS" --models "$M" --readouts $READOUTS \
            --perturbations none $PERTS --p $MATCHED_P --folds $MATCHED_FOLDS
    done
done
