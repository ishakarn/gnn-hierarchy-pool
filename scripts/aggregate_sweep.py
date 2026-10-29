"""Collect the hyperparameter sweep into sweep_results.csv and pick each readout's config (sweep_selected.csv).

Follows docs/hparam_sweep_protocol.ipynb: for every (dataset, model, readout), average the
validation metric of each (layers, dropout) config over its seeds, keep configs with at least
--min_seeds valid seeds, pick the best (ties within 0.001: fewer layers, then lower dropout)
and report that config's test metric as mean ± std and 95% CI over seeds. Test is never used
to choose anything.

Usage:
    python scripts/aggregate_sweep.py --results_dir results/sweep
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_sweep import LAYERS, DROPOUTS, SEEDS


TIE_TOL       = 0.001                                   # configs this close on mean val count as tied
EXPECTED_RUNS = len(LAYERS) * len(DROPOUTS) * len(SEEDS)  # 150 per (dataset, model, readout)


def _primary_metric(task_type: str) -> tuple:
    """Return (metric name, higher_is_better) for a task type."""
    if task_type == 'binary':
        return 'auc', True
    return 'rmse', False


def _ci95(values: np.ndarray) -> float:
    """Half-width of the 95% t-interval for the mean (NaN with fewer than 2 values)."""
    n = len(values)
    if n < 2:
        return float('nan')
    return float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def select_configs(df: pd.DataFrame, min_seeds: int = 4) -> pd.DataFrame:
    """One row per (dataset, model, readout): the config chosen on mean validation over seeds.

    Args:
        df: one row per run (sweep_results.csv).
        min_seeds: valid (non-NaN) seeds a config needs to be eligible.

    Returns:
        DataFrame with the chosen layers/dropout, its test mean/std/CI/n and audit columns.
    """
    rows = []
    for (dataset, model, readout), group in df.groupby(['dataset', 'model', 'readout']):
        metric, higher = _primary_metric(group['task_type'].iloc[0])
        val_col, test_col = f'val_{metric}', f'test_{metric}'

        # Mean val per config over its valid seeds; NaN seeds (one-class val) are not counted
        per_config = (group.groupby(['num_layers', 'dropout'])[val_col]
                      .agg(val_mean='mean', n_valid='count').reset_index())
        eligible = per_config[per_config['n_valid'] >= min_seeds]
        if eligible.empty:
            continue

        best_val = eligible['val_mean'].max() if higher else eligible['val_mean'].min()
        gap      = best_val - eligible['val_mean'] if higher else eligible['val_mean'] - best_val
        chosen   = eligible[gap <= TIE_TOL].sort_values(['num_layers', 'dropout']).iloc[0]

        runs = group[(group['num_layers'] == chosen['num_layers'])
                     & (group['dropout'] == chosen['dropout'])]
        test = runs[test_col].dropna().to_numpy()
        rows.append({
            'dataset':         dataset,
            'model':           model,
            'readout':         readout,
            'metric':          metric,
            'num_layers':      int(chosen['num_layers']),
            'dropout':         float(chosen['dropout']),
            'val_mean':        chosen['val_mean'],
            'test_mean':       test.mean() if len(test) else float('nan'),
            'test_std':        test.std(ddof=1) if len(test) > 1 else float('nan'),
            'test_ci95':       _ci95(test),
            'n':               len(test),
            'best_epoch_mean': runs['best_epoch'].mean(),
            'n_runs':          len(group),
            # Optimum may lie outside the grid when the chosen config sits on its edge
            'at_grid_edge':    (chosen['num_layers'] in (min(LAYERS), max(LAYERS))
                                or chosen['dropout'] in (min(DROPOUTS), max(DROPOUTS))),
        })
    if not rows:
        return pd.DataFrame()

    # Best readout per (dataset, model), compared on the selected configs' mean val
    selected = pd.DataFrame(rows)
    selected['best_readout'] = False
    for _, group in selected.groupby(['dataset', 'model']):
        best_idx = group['val_mean'].idxmax() if group['metric'].iloc[0] == 'auc' else group['val_mean'].idxmin()
        selected.loc[best_idx, 'best_readout'] = True
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--results_dir', default='./results/sweep')
    p.add_argument('--min_seeds',   type=int, default=4,
                   help='Valid seeds a config needs to be eligible (protocol: 4 of 5)')
    args = p.parse_args()

    # results/sweep/<DATASET>/<model>/<readout>/L<l>_do<d>/*_metrics.json
    files = sorted(glob.glob(os.path.join(args.results_dir, '*', '*', '*', '*', '*_metrics.json')))
    if not files:
        print(f"No *_metrics.json files under {args.results_dir}/<DATASET>/<model>/<readout>/<config>/")
        return

    rows = []
    for f in files:
        with open(f) as fh:
            rows.append(json.load(fh))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, 'sweep_results.csv'), index=False)

    selected = select_configs(df, min_seeds=args.min_seeds)
    selected.to_csv(os.path.join(args.results_dir, 'sweep_selected.csv'), index=False)
    print(f"{len(df)} runs → {args.results_dir}/sweep_results.csv, sweep_selected.csv\n")

    # ---- Checks (protocol: after the runs finish) ----
    counts     = df.groupby(['dataset', 'model', 'readout']).size()
    incomplete = counts[counts < EXPECTED_RUNS]
    if len(incomplete):
        print(f"Incomplete ({len(incomplete)} of {len(counts)} have < {EXPECTED_RUNS} runs):")
        for (dataset, model, readout), n in incomplete.items():
            print(f"  {dataset}/{model}/{readout}: {n}")
    print(f"Runs stopped by the epoch cap (best_epoch > 250): {(df['best_epoch'] > 250).mean():.1%}")
    for dataset, group in df.groupby('dataset'):
        metric, _ = _primary_metric(group['task_type'].iloc[0])
        n_nan = group[[f'val_{metric}', f'test_{metric}']].isna().sum()
        if n_nan.any():
            print(f"  {dataset}: NaN val_{metric} {n_nan.iloc[0]}, test_{metric} {n_nan.iloc[1]}")

    # ---- Headline: test mean ± std (layers, dropout) ----
    if selected.empty:
        print(f"\nNo config has {args.min_seeds} valid seeds yet.")
        return
    print()
    for _, r in selected.iterrows():
        flags = (" *best*" if r['best_readout'] else "") + (" [grid edge]" if r['at_grid_edge'] else "")
        print(f"  {r['dataset']:8s} | {r['model']:8s} | {r['readout']:9s} | "
              f"test_{r['metric']} {r['test_mean']:.4f} ± {r['test_std']:.4f} "
              f"(L{r['num_layers']}, do{r['dropout']}) | n={r['n']}{flags}")


if __name__ == '__main__':
    main()
    