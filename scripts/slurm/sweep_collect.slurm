#!/bin/bash
#SBATCH --job-name=sweep_collect
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
source /etc/profile
module load python/3.12.3
source "${VENV:-$HOME/venvs/sage}/bin/activate"
set -u
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

python scripts/sweep_jobs.py collect "$1"
