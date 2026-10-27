#!/bin/bash
# After grid jobs finish, run this CPU-only job to aggregate *_metrics.json
# into grid_results.csv / grid_summary.csv. Submit from the repo root:
#   sbatch --dependency=afterany:<JOB_IDS> scripts/slurm/aggregate_results.slurm [results_dir]
# (Cheap enough to run directly: python scripts/aggregate_results.py --results_dir <dir>)

#SBATCH --job-name=gnn_aggregate
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --output=logs/aggregate_%j.out

set -eo pipefail
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate graph-pooling
set -u

cd "$SLURM_SUBMIT_DIR"
python scripts/aggregate_results.py --results_dir "${1:-results}"
