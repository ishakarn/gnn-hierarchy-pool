#!/bin/bash
#SBATCH --job-name=jknet_metadata
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
source /etc/profile
module load python/3.12.3
source "$VENV/bin/activate"
set -u
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python tests/check_spectra_metadata.py
for file in scripts/slurm/run_sweep_array.slurm scripts/slurm/sweep_prepare.slurm scripts/slurm/sweep_collect.slurm scripts/slurm/run_grid_array.slurm scripts/slurm/submit_sweep.sh; do
    bash -n "$file"
done
