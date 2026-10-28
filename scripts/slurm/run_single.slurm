#!/bin/bash
# Submit one experiment (dataset/model/readout/seed) as a single GPU job.
# Usage:
#   sbatch scripts/slurm/run_single.slurm \
#          --config configs/bbbp.yaml --model gcn --readout mean --seed 0
#
# Must be submitted from the repo root.

#SBATCH --job-name=gnn_readout
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate graph-pooling

cd "$SLURM_SUBMIT_DIR"

python scripts/run_single.py "$@"
