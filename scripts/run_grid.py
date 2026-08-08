"""Grid search: run all (dataset, model, readout, seed) combinations."""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import argparse
import json
import subprocess
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


DATASET_TO_CONFIG = {
    'BBBP': 'configs/bbbp.yaml',
    'BACE': 'configs/bace.yaml',
    # Add ClinTox / HIV here when configs are ready:
    # 'CLINTOX': 'configs/clintox.yaml',
    # 'HIV':     'configs/hiv.yaml',
}


def _run_single(dataset, model, readout, seed, output_dir, extra_args):
    """Invoke run_single.py as a subprocess and stream its stdout."""
    config = DATASET_TO_CONFIG[dataset.upper()]
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'run_single.py'),
        '--config',     config,
        '--model',      model,
        '--readout',    readout,
        '--seed',       str(seed),
        '--output_dir', output_dir,
    ] + extra_args

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {dataset}/{model}/{readout}/seed{seed} — exit code {result.returncode}")
        return False
    return True


def _collect_results(output_dir, datasets, models, readouts, seeds) -> pd.DataFrame:
    rows = []
    for dataset, model, readout, seed in product(datasets, models, readouts, seeds):
        run_name     = f"{dataset.upper()}_{model}_{readout}_seed{seed}"
        metrics_path = os.path.join(output_dir, f"{run_name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                rows.append(json.load(f))
        else:
            rows.append({
                'dataset': dataset.upper(), 'model': model,
                'readout': readout, 'seed': seed,
                'train_auc': None, 'val_auc': None,
                'test_auc': None, 'best_epoch': None,
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(
        description="Run a grid of GNN/readout experiments on MoleculeNet datasets."
    )
    p.add_argument('--datasets', nargs='+',
                   default=['BBBP', 'BACE'],
                   choices=list(DATASET_TO_CONFIG.keys()))
    p.add_argument('--models',   nargs='+',
                   default=['gcn', 'gin', 'gat', 'graphsage', 'gcnp'],
                   choices=['gcn', 'gin', 'gat', 'graphsage', 'gcnp'])
    p.add_argument('--readouts', nargs='+',
                   default=['mean', 'sum', 'max', 'attention', 'selfattention'],
                   choices=['mean', 'sum', 'max', 'attention', 'selfattention'])
    p.add_argument('--seeds',    nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--output_dir', default='./results')
    p.add_argument('--epochs',   type=int, default=None,
                   help='Override epochs in each config (e.g. for smoke test)')
    args = p.parse_args()

    extra_args = ['--epochs', str(args.epochs)] if args.epochs else []

    datasets  = [d.upper() for d in args.datasets]
    combos    = list(product(datasets, args.models, args.readouts, args.seeds))
    n_total   = len(combos)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Grid: {len(datasets)} dataset(s) × {len(args.models)} model(s) × "
          f"{len(args.readouts)} readout(s) × {len(args.seeds)} seed(s) = {n_total} runs\n")

    failed = []
    for i, (dataset, model, readout, seed) in enumerate(combos, 1):
        label = f"{dataset}/{model}/{readout}/seed{seed}"
        print(f"\n[{i}/{n_total}] {label}")
        ok = _run_single(dataset, model, readout, seed, args.output_dir, extra_args)
        if not ok:
            failed.append(label)

    # ---- Collect per-run results ----
    df = _collect_results(args.output_dir, datasets, args.models, args.readouts, args.seeds)
    per_run_csv = os.path.join(args.output_dir, 'grid_results.csv')
    df.to_csv(per_run_csv, index=False)
    print(f"\nPer-run results  → {per_run_csv}")

    # ---- Aggregate summary ----
    numeric = ['train_auc', 'val_auc', 'test_auc']
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')
    summary = (
        df.groupby(['dataset', 'model', 'readout'])[numeric]
        .agg(['mean', 'std'])
        .round(4)
    )
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary_csv = os.path.join(args.output_dir, 'grid_summary.csv')
    summary.to_csv(summary_csv)
    print(f"Aggregate summary → {summary_csv}\n")
    print(summary.to_string())

    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for f in failed:
            print(f"  {f}")


if __name__ == '__main__':
    main()
