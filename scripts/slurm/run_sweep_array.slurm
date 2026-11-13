#!/bin/bash
#SBATCH --job-name=jknet_sweep
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -eo pipefail
source /etc/profile
module load python/3.12.3
source "${VENV:-$HOME/venvs/sage}/bin/activate"
set -u
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

python -c 'import torch; assert torch.cuda.is_available(), "GPU unavailable"'
python scripts/sweep_jobs.py run "$1" "$SLURM_ARRAY_TASK_ID"
