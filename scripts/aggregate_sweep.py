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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_sweep import LAYERS, DROPOUTS, SEEDS
from src.results import (
    METRIC_LABELS, READOUT_ORDER, model_label, ordered, readout_color, readout_label,
)


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


def _pick_config(configs: pd.DataFrame, higher: bool) -> pd.Series:
    """Best row of `configs` on val_mean; ties within TIE_TOL go to fewer layers, then lower dropout."""
    best_val = configs['val_mean'].max() if higher else configs['val_mean'].min()
    gap      = best_val - configs['val_mean'] if higher else configs['val_mean'] - best_val
    return configs[gap <= TIE_TOL].sort_values(['num_layers', 'dropout']).iloc[0]


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

        chosen = _pick_config(eligible, higher)

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


def plot_sweep(df: pd.DataFrame, selected: pd.DataFrame, out_dir: str, min_seeds: int = 4):
    """Save one figure per (dataset, model): test metric vs depth, and test metric per readout.

    Left panel: for each readout and depth, the dropout picked on mean val at that depth (same
    rule as select_configs), plotted as test mean ± std over seeds. Right panel: each readout's
    selected config from select_configs, test mean ± std. AUC uses a fixed 0.5-1 y-axis
    (0.5 = chance) so small gaps between readouts are not blown up.
    """
    os.makedirs(out_dir, exist_ok=True)
    for (dataset, model), group in df.groupby(['dataset', 'model']):
        metric, higher = _primary_metric(group['task_type'].iloc[0])
        val_col, test_col = f'val_{metric}', f'test_{metric}'
        readouts = ordered(group['readout'].unique(), READOUT_ORDER)
        sel      = selected[(selected['dataset'] == dataset) & (selected['model'] == model)]

        fig, (ax_depth, ax_readout) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for i, readout in enumerate(readouts):
            runs = group[group['readout'] == readout]
            per_config = (runs.groupby(['num_layers', 'dropout'])[val_col]
                          .agg(val_mean='mean', n_valid='count').reset_index())
            per_config = per_config[per_config['n_valid'] >= min_seeds]

            # Left: best dropout at each depth, scored on test
            depths, means, stds = [], [], []
            for layers, configs in per_config.groupby('num_layers'):
                dropout = _pick_config(configs, higher)['dropout']
                test    = runs[(runs['num_layers'] == layers) & (runs['dropout'] == dropout)][test_col].dropna()
                depths.append(layers)
                means.append(test.mean())
                stds.append(test.std(ddof=1))
            color = readout_color(readout, i)
            ax_depth.errorbar(depths, means, yerr=stds, marker='o', capsize=3,
                              color=color, label=readout_label(readout))

            # Right: the selected config of this readout
            row = sel[sel['readout'] == readout]
            if len(row):
                ax_readout.errorbar(i, row['test_mean'].iloc[0], yerr=row['test_std'].iloc[0],
                                    fmt='o', capsize=5, color=color)

        label = f"Test {METRIC_LABELS.get(metric, metric)}"
        ax_depth.set_xticks(sorted(group['num_layers'].unique()))
        ax_depth.set_xlabel('Number of layers')
        ax_depth.set_ylabel(label)
        ax_depth.legend(fontsize=8)
        ax_readout.set_xticks(range(len(readouts)), [readout_label(r) for r in readouts])
        ax_readout.set_xlim(-0.5, len(readouts) - 0.5)
        ax_readout.set_xlabel('Readout (selected config)')
        if metric == 'auc':
            ax_depth.set_ylim(0.7, 0.9)
        fig.suptitle(f"{dataset} | {model_label(model)}")

        path = os.path.join(out_dir, f"{dataset}_{model}_sweep.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--results_dir', default='./results/sweep')
    p.add_argument('--min_seeds',   type=int, default=4,
                   help='Valid seeds a config needs to be eligible (protocol: 4 of 5)')
    p.add_argument('--plot',        action='store_true',
                   help='Also save the depth / readout figures to <results_dir>/plots/')
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

    if args.plot:
        print()
        plot_sweep(df, selected, os.path.join(args.results_dir, 'plots'), min_seeds=args.min_seeds)


if __name__ == '__main__':
    main()