#!/bin/bash
# Base training runs for ONE dataset: GIN/GINE x all readouts x seeds, run in
# sequence via run_grid.py into results/base/<DATASET>/. Resumable: completed
# runs are skipped, so a preempted/requeued job picks up where it stopped.
#
#   sbatch scripts/slurm/run_base.slurm ESOL                      # default grid
#   sbatch scripts/slurm/run_base.slurm BBBP --readouts mean sum   # extra args -> run_grid.py
#
# Usually submitted for all datasets by scripts/slurm/submit_base.sh.

#SBATCH --job-name=gnn_base
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
hostname
date
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate graph-pooling
set -u
module list
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
which python && python --version
nvidia-smi --query-gpu=name,memory.total --format=csv

DATASET=$1
shift
python scripts/run_grid.py \
    --datasets "$DATASET" \
    --models gin gine \
    --readouts mean sum max attention selfattention \
    --seeds 0 1 2 \
    --output_dir "results/base/$DATASET" \
    --skip_existing \
    "$@"
date
