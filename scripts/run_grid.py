"""Grid search: run all (dataset, model, readout, seed) combinations.

With --split spectra the grid also loops over SPECTRA parameters (--sps) and
split replicates (--split_idxs).
"""
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

from aggregate_results import grid_summary


DATASET_TO_CONFIG = {
    'BBBP': 'configs/bbbp.yaml',
    'BACE': 'configs/bace.yaml',
    'CLINTOX': 'configs/clintox.yaml',
    # Regression
    'ESOL':     'configs/esol.yaml',
    'FREESOLV': 'configs/freesolv.yaml',
    'LIPO':     'configs/lipo.yaml',
    # Add HIV here when a config is ready:
    # 'HIV':     'configs/hiv.yaml',
}


from src.run_identity import run_name as _run_name

_RUNNERS = {}


def _run_single(dataset, model, readout, seed, output_dir, extra_args,
                sp=None, split_idx=None, config=None, in_process=False,
                jk_mode='cat', graph_cache_dir='./data/cache/graphs'):
    """Invoke run_single.py as a subprocess and stream its stdout."""
    config = config or DATASET_TO_CONFIG[dataset.upper()]
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'run_single.py'),
        '--config',     config,
        '--model',      model,
        '--readout',    readout,
        '--seed',       str(seed),
        '--output_dir', output_dir,
    ] + extra_args
    if sp is not None:
        cmd += ['--split', 'spectra', '--sp', f"{sp:.2f}", '--split_idx', str(split_idx)]

    if model == 'jknet':
        cmd += ['--jk_mode', jk_mode]
    cmd += ['--graph_cache_dir', graph_cache_dir]
    if in_process:
        from src.experiment_runner import ExperimentRunner
        if graph_cache_dir not in _RUNNERS:
            _RUNNERS[graph_cache_dir] = ExperimentRunner(graph_cache_dir)
        return _RUNNERS[graph_cache_dir].run(cmd[2:])
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {_run_name(dataset, model, readout, seed, sp, split_idx, jk_mode=jk_mode)}"
              f" — exit code {result.returncode}")
        return False
    return True


def _collect_results(output_dir, combos, jk_mode='cat') -> pd.DataFrame:
    rows = []
    for dataset, model, readout, seed, sp, split_idx in combos:
        run_name     = _run_name(dataset, model, readout, seed, sp, split_idx, jk_mode=jk_mode)
        metrics_path = os.path.join(output_dir, f"{run_name}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                rows.append(json.load(f))
        else:
            rows.append({
                'dataset': dataset.upper(), 'model': model,
                'readout': readout, 'seed': seed,
                'sp': sp, 'split_idx': split_idx, 'status': 'missing',
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
                   default=['gin', 'gine'],
                   choices=['gcn', 'gin', 'gine', 'gat', 'graphsage', 'gcnp', 'attentivefp', 'jknet', 'chemprop'])
    p.add_argument('--readouts', nargs='+',
                   default=['mean', 'sum', 'max', 'attention', 'multihead'],
                   choices=['mean', 'sum', 'max', 'attention', 'multihead', 'selfattention', 'afp'])
    p.add_argument('--seeds',    nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--output_dir', default='./results')
    p.add_argument('--epochs',   type=int, default=None,
                   help='Override epochs in each config (e.g. for smoke test)')
    p.add_argument('--skip_existing', action='store_true',
                   help='Skip runs whose metrics JSON already exists (resume after preemption)')

    # ---- SPECTRA splits ----
    p.add_argument('--split', choices=['scaffold', 'spectra'], default='scaffold')
    p.add_argument('--sps', nargs='+', type=float,
                   default=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                   help='SPECTRA parameters to run (only with --split spectra)')
    p.add_argument('--split_idxs', nargs='+', type=int, default=[0, 1, 2],
                   help='SPECTRA split replicates to run (only with --split spectra)')
    p.add_argument('--jk_mode', choices=['cat', 'max', 'lstm'], default='cat')
    p.add_argument('--fresh_process', action='store_true',
                   help='Run each fit in a subprocess (default: reuse process and graph cache)')
    p.add_argument('--graph_cache_dir', default='./data/cache/graphs')
    args = p.parse_args()
    jk_mode = args.jk_mode

    extra_args = ['--epochs', str(args.epochs)] if args.epochs else []

    datasets  = [d.upper() for d in args.datasets]
    if args.split == 'spectra':
        sps, split_idxs = args.sps, args.split_idxs
    else:
        sps, split_idxs = [None], [None]
    combos    = list(product(datasets, args.models, args.readouts, args.seeds,
                             sps, split_idxs))
    n_total   = len(combos)

    os.makedirs(args.output_dir, exist_ok=True)
    grid_desc = (f"{len(datasets)} dataset(s) × {len(args.models)} model(s) × "
                 f"{len(args.readouts)} readout(s) × {len(args.seeds)} seed(s)")
    if args.split == 'spectra':
        grid_desc += f" × {len(sps)} SP value(s) × {len(split_idxs)} split(s)"
    print(f"Grid: {grid_desc} = {n_total} runs\n")

    failed = []
    for i, (dataset, model, readout, seed, sp, split_idx) in enumerate(combos, 1):
        run_name = _run_name(dataset, model, readout, seed, sp, split_idx, jk_mode=jk_mode)
        print(f"\n[{i}/{n_total}] {run_name}")
        if args.skip_existing and os.path.exists(os.path.join(args.output_dir, f"{run_name}_metrics.json")):
            print("  [skip] metrics already exist")
            continue
        ok = _run_single(dataset, model, readout, seed, args.output_dir, extra_args,
                         sp=sp, split_idx=split_idx, in_process=not args.fresh_process,
                         jk_mode=jk_mode, graph_cache_dir=args.graph_cache_dir)
        if not ok:
            failed.append(run_name)

    # ---- Collect per-run results ----
    df = _collect_results(args.output_dir, combos, jk_mode=jk_mode)
    per_run_csv = os.path.join(args.output_dir, 'grid_results.csv')
    df.to_csv(per_run_csv, index=False)
    print(f"\nPer-run results  → {per_run_csv}")

    # ---- Aggregate summary ----
    summary = grid_summary(df)
    summary_csv = os.path.join(args.output_dir, 'grid_summary.csv')
    summary.to_csv(summary_csv)
    print(f"Aggregate summary → {summary_csv}\n")
    print(summary.to_string())

    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == '__main__':
    main()
