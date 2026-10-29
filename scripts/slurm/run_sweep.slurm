#!/bin/bash
# Hyperparameter sweep for ONE (dataset, model, readout): 30 configs x 5 seeds run in
# sequence via run_sweep.py into results/sweep/. Resumable: completed runs are skipped,
# so a preempted/requeued or timed-out job picks up where it stopped when resubmitted.
#
#   sbatch scripts/slurm/run_sweep.slurm BBBP chemprop mean
#   sbatch scripts/slurm/run_sweep.slurm BBBP gine sum --layers 2 --epochs 2   # extra args -> run_sweep.py
#
# Usually submitted for all datasets x readouts of one model by scripts/slurm/submit_sweep.sh.
# Uses the venv at $VENV (default ~/venvs/sage).

#SBATCH --job-name=gnn_sweep
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
hostname
date
module load python/3.12.3
source "${VENV:-$HOME/venvs/sage}/bin/activate"
set -u
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
which python && python --version
nvidia-smi --query-gpu=name,memory.total --format=csv

DATASET=$1
MODEL=$2
READOUT=$3
shift 3
python scripts/run_sweep.py \
    --dataset "$DATASET" \
    --model   "$MODEL" \
    --readout "$READOUT" \
    "$@"
date
