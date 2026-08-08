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
df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')
summary = (
    df.groupby(['dataset', 'model', 'readout'])[numeric]
    .agg(['mean', 'std'])
    .round(4)
)
summary.columns = [f"{c}_{s}" for c, s in summary.columns]
summary.to_csv(os.path.join(results_dir, 'grid_summary.csv'))
print(summary.to_string())
EOF
