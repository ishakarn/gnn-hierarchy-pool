#!/bin/bash
# One ModeTrace job. First argument picks the protocol (matched | ood); the rest
# is passed to scripts/modetrace/run_<protocol>.py. Submit from the repo root,
# after creating logs/ (SLURM opens the log files before the script runs):
#
#   mkdir -p logs
#   sbatch scripts/slurm/run_modetrace.slurm matched \
#       --datasets BACE --models gin gine --readouts mean \
#       --perturbations none GNFS GEFS JGFS --folds 0 --epochs 2      # smoke test
#
#   sbatch scripts/slurm/run_modetrace.slurm ood \
#       --datasets ESOL --models gin gine --perturbations WNFS GNFS WEFS GEFS JGFS
#
# Runs resume: completed folds are skipped, so a preempted job can be resubmitted.

#SBATCH --job-name=modetrace
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
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

PROTOCOL=$1
shift
python scripts/modetrace/run_${PROTOCOL}.py "$@"
