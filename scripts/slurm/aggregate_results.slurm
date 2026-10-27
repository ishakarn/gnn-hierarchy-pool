#!/bin/bash
# After all grid array jobs finish, run this CPU-only job to aggregate results.
# Submit with:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/slurm/aggregate_results.slurm

#SBATCH --job-name=gnn_aggregate
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --output=logs/aggregate_%j.out

module load conda/latest
conda activate graph-pooling

cd /work/pi_annagreen_umass_edu/Isha/graph-pooling/gnn_readout_bio

python - <<'EOF'
import os, json, pandas as pd
from itertools import product

results_dir = 'results'
rows = []
for f in os.listdir(results_dir):
    if f.endswith('_metrics.json'):
        with open(os.path.join(results_dir, f)) as fh:
            rows.append(json.load(fh))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(results_dir, 'grid_results.csv'), index=False)

numeric = ['train_auc', 'val_auc', 'test_auc']
for c in numeric:
    if c not in df.columns:
        df[c] = None
df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')

# SPECTRA runs: also group by SP, and report skipped splits
group_cols = ['dataset', 'model', 'readout']
if 'sp' in df.columns and df['sp'].notna().any():
    group_cols.append('sp')
if 'status' in df.columns:
    print(f"Skipped runs: {(df['status'] == 'skipped').sum()}")
summary = (
    df.groupby(group_cols, dropna=False)[numeric]
    .agg(['mean', 'std'])
    .round(4)
)
summary.columns = [f"{c}_{s}" for c, s in summary.columns]
summary.to_csv(os.path.join(results_dir, 'grid_summary.csv'))
print(summary.to_string())
EOF
