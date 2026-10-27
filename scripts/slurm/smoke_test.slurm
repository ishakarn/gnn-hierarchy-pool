#!/bin/bash
# Smoke test on a GPU: every dataset x one model x 2 epochs, plus a tiny
# ModeTrace matched + OOD run. Also processes the MoleculeNet raw files into
# data/*/processed once, before parallel jobs start. Submit from the repo root:
#   mkdir -p logs && sbatch scripts/slurm/smoke_test.slurm

#SBATCH --job-name=gnn_smoke
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:45:00
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

OUT=results/smoke
python scripts/run_grid.py --datasets BBBP BACE CLINTOX ESOL FREESOLV LIPO \
    --models gin gine --readouts mean selfattention --seeds 0 --epochs 2 --output_dir "$OUT/base"

python scripts/modetrace/run_matched.py --datasets BACE ESOL --models gin gine \
    --perturbations none GNFS GEFS --p 1.0 --folds 0 --epochs 2 --output_dir "$OUT/modetrace"
python scripts/modetrace/run_ood.py --datasets CLINTOX ESOL --models gine \
    --perturbations GNFS JGFS --p 1.0 --folds 0 --epochs 2 --output_dir "$OUT/modetrace"
python scripts/modetrace/summarize.py --results_dir "$OUT/modetrace"

date
echo "SMOKE TEST OK"
