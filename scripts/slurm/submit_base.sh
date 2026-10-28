#!/bin/bash
# Submit one base-training job per dataset, then a CPU job that aggregates
# everything into results/base/grid_results.csv once they all finish.
#
#   bash scripts/slurm/submit_base.sh              # all datasets
#   bash scripts/slurm/submit_base.sh ESOL LIPO    # a subset
#   bash scripts/slurm/submit_base.sh --dry-run    # print the sbatch commands only
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; shift; fi
DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(BBBP BACE CLINTOX ESOL FREESOLV LIPO)

JOB_IDS=()
for DS in "${DATASETS[@]}"; do
    CMD=(sbatch --parsable --job-name="base_${DS,,}" scripts/slurm/run_base.slurm "$DS")
    if $DRY_RUN; then echo "${CMD[*]}"; continue; fi
    ID=$("${CMD[@]}")
    JOB_IDS+=("$ID")
    echo "submitted $DS -> job $ID"
done

if ! $DRY_RUN && [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    DEP=$(IFS=:; echo "${JOB_IDS[*]}")
    AGG=$(sbatch --parsable --dependency=afterany:"$DEP" scripts/slurm/aggregate_results.slurm results/base)
    echo "aggregate job $AGG (runs after $DEP)"
fi
