"""Collect the hyperparameter sweep into sweep_results.csv and pick each readout's config (sweep_selected.csv).

Follows docs/hparam_sweep_protocol.ipynb: for every (dataset, model, readout), average the
validation metric of each (layers, dropout, hidden_dim) config over its seeds, keep configs
with at least --min_seeds valid seeds, pick the best (ties within 0.001: fewer layers, then
lower dropout, then smaller hidden_dim)
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
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_sweep import LAYERS, DROPOUTS, HIDDEN_DIMS, SEEDS
from src.results import (
    METRIC_LABELS, READOUT_ORDER, model_label, ordered, readout_color, readout_label,
)


TIE_TOL       = 0.001                                   # configs this close on mean val count as tied
EXPECTED_RUNS = len(LAYERS) * len(DROPOUTS) * len(HIDDEN_DIMS) * len(SEEDS)  # 630 per (dataset, model, readout)


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
    """Best row of `configs` on val_mean.

    Ties within TIE_TOL go to fewer layers, then lower dropout, then smaller hidden_dim.
    """
    best_val = configs['val_mean'].max() if higher else configs['val_mean'].min()
    gap      = best_val - configs['val_mean'] if higher else configs['val_mean'] - best_val
    return configs[gap <= TIE_TOL].sort_values(['num_layers', 'dropout', 'hidden_dim']).iloc[0]


def select_configs(df: pd.DataFrame, min_seeds: int = 4) -> pd.DataFrame:
    """One row per (dataset, model, readout): the config chosen on mean validation over seeds.

    Args:
        df: one row per run (sweep_results.csv).
        min_seeds: valid (non-NaN) seeds a config needs to be eligible.

    Returns:
        DataFrame with the chosen layers/dropout/hidden_dim, its test mean/std/CI/n
        and audit columns.
    """
    rows = []
    for (dataset, model, readout), group in df.groupby(['dataset', 'model', 'readout']):
        metric, higher = _primary_metric(group['task_type'].iloc[0])
        val_col, test_col = f'val_{metric}', f'test_{metric}'

        # Mean val per config over its valid seeds; NaN seeds (one-class val) are not counted
        per_config = (group.groupby(['num_layers', 'dropout', 'hidden_dim'])[val_col]
                      .agg(val_mean='mean', n_valid='count').reset_index())
        eligible = per_config[per_config['n_valid'] >= min_seeds]
        if eligible.empty:
            continue

        chosen = _pick_config(eligible, higher)

        runs = group[(group['num_layers'] == chosen['num_layers'])
                & (group['dropout'] == chosen['dropout'])
                & (group['hidden_dim'] == chosen['hidden_dim'])]
        test = runs[test_col].dropna().to_numpy()
        rows.append({
            'dataset':         dataset,
            'model':           model,
            'readout':         readout,
            'metric':          metric,
            'num_layers':      int(chosen['num_layers']),
            'dropout':         float(chosen['dropout']),
            'hidden_dim':      int(chosen['hidden_dim']),
            'val_mean':        chosen['val_mean'],
            'test_mean':       test.mean() if len(test) else float('nan'),
            'test_std':        test.std(ddof=1) if len(test) > 1 else float('nan'),
            'test_ci95':       _ci95(test),
            'n':               len(test),
            'best_epoch_mean': runs['best_epoch'].mean(),
            'n_runs':          len(group),
            # Optimum may lie outside the grid when the chosen config sits on its edge
            'at_grid_edge':    (chosen['num_layers'] in (min(LAYERS), max(LAYERS))
                                or chosen['dropout'] in (min(DROPOUTS), max(DROPOUTS))
                                or chosen['hidden_dim'] in (min(HIDDEN_DIMS), max(HIDDEN_DIMS))),
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
            per_config = (runs.groupby(['num_layers', 'dropout', 'hidden_dim'])[val_col]
                          .agg(val_mean='mean', n_valid='count').reset_index())
            per_config = per_config[per_config['n_valid'] >= min_seeds]

            # Left: best dropout at each depth, scored on test
            depths, means, stds = [], [], []
            for layers, configs in per_config.groupby('num_layers'):
                best = _pick_config(configs, higher)
                test = runs[(runs['num_layers'] == layers)
                            & (runs['dropout'] == best['dropout'])
                            & (runs['hidden_dim'] == best['hidden_dim'])][test_col].dropna()
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
            ax_depth.set_ylim(0.7, 1.0)
        fig.suptitle(f"{dataset} | {model_label(model)}")

        path = os.path.join(out_dir, f"{dataset}_{model}_sweep.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")

def plot_heatmaps(df: pd.DataFrame, selected: pd.DataFrame, out_dir: str, min_seeds: int = 4):
    """Save one figure per (dataset, model): a layers x dropout heatmap of the test metric per readout.

    Colour is the mean test metric over seeds, on one scale for the whole figure so readouts
    can be compared against each other. The cell chosen on validation is outlined, so a
    selected cell that is not the best-coloured one shows exactly where validation and test
    disagree. Configs with fewer than `min_seeds` valid seeds are left blank.
    """
    os.makedirs(out_dir, exist_ok=True)
    for (dataset, model, hidden_dim), group in df.groupby(['dataset', 'model', 'hidden_dim']):
        metric, higher = _primary_metric(group['task_type'].iloc[0])
        test_col = f'test_{metric}'
        readouts = ordered(group['readout'].unique(), READOUT_ORDER)
        layers   = sorted(group['num_layers'].unique())
        dropouts = sorted(group['dropout'].unique())
        sel = selected[(selected['dataset'] == dataset)
                       & (selected['model'] == model)
                       & (selected['hidden_dim'] == hidden_dim)]

        # Mean test metric per cell; cells short of min_seeds are dropped, not averaged
        cells = (group.groupby(['readout', 'num_layers', 'dropout'])[test_col]
                 .agg(mean='mean', n_valid='count').reset_index())
        cells = cells[cells['n_valid'] >= min_seeds]
        if cells.empty:
            continue

        # Percentile limits: one collapsed config must not flatten the whole scale
        vmin, vmax = cells['mean'].quantile([0.02, 0.98])
        cmap = 'Blues' if higher else 'Blues_r'   # always: better = darker

        fig, axes = plt.subplots(1, len(readouts), figsize=(3.6 * len(readouts), 4.2),
                                 sharey=True, squeeze=False)
        for ax, readout in zip(axes[0], readouts):
            # [dropouts, layers]; NaN cells render as blank
            grid = np.full((len(dropouts), len(layers)), np.nan)
            for _, c in cells[cells['readout'] == readout].iterrows():
                grid[dropouts.index(c['dropout']), layers.index(c['num_layers'])] = c['mean']

            im = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

            row = sel[sel['readout'] == readout]
            if len(row):
                r = row.iloc[0]
                ax.add_patch(plt.Rectangle(
                    (layers.index(r['num_layers']) - 0.5, dropouts.index(r['dropout']) - 0.5),
                    1, 1, fill=False, edgecolor='red', linewidth=2,
                ))

            # Best cell on test: where validation would have landed with hindsight
            sub = cells[cells['readout'] == readout]
            if len(sub):
                b = sub.loc[sub['mean'].idxmax() if higher else sub['mean'].idxmin()]
                ax.add_patch(plt.Rectangle(
                    (layers.index(b['num_layers']) - 0.5, dropouts.index(b['dropout']) - 0.5),
                    1, 1, fill=False, edgecolor='limegreen', linewidth=2, linestyle='--',
                ))

            ax.set_xticks(range(len(layers)), layers)
            ax.set_xlabel('Layers')
            ax.set_title(readout_label(readout), fontsize=10)

        axes[0][0].set_yticks(range(len(dropouts)), dropouts)
        axes[0][0].set_ylabel('Dropout')
        fig.colorbar(im, ax=axes[0], label=f"Test {METRIC_LABELS.get(metric, metric)}")
        fig.suptitle(f"{dataset} | {model_label(model)} | width {hidden_dim}   "f"(red = chosen on validation, green dashed = best on test)")

        path = os.path.join(out_dir, f"{dataset}_{model}_h{hidden_dim}_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved {path}")

# SCATTER_HPARAMS = ['hidden_dim', 'num_layers', 'dropout', 'readout']
SCATTER_HPARAMS = ['num_layers']
HPARAM_LABELS = {'num_layers': 'n layers', 'dropout': 'dropout', 'hidden_dim': 'width', 'readout': 'readout'}
BASELINE_READOUT = 'mean'

def plot_architectures(df: pd.DataFrame, out_dir: str):
    """Save one figure per (dataset, model): parameters vs test metric, coloured by hyperparameter.

    Every run is its own dot, so the vertical spread at a given parameter count is the
    seed-to-seed spread — no averaging and no error bars. The narrow panel on the right
    boxes the same runs by level of that hyperparameter. Styled after the architecture
    figures in distill.pub/2021/gnn-intro.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for (dataset, model), group in df.groupby(['dataset', 'model']):
        metric, _ = _primary_metric(group['task_type'].iloc[0])
        test_col  = f'test_{metric}'
        runs      = group.dropna(subset=[test_col, 'n_params'])
        if runs.empty:
            continue

        for hparam in SCATTER_HPARAMS:
            hlabel = HPARAM_LABELS.get(hparam, hparam)
            levels = (ordered(runs[hparam].unique(), READOUT_ORDER)
                      if hparam == 'readout' else sorted(runs[hparam].unique()))

            fig, (ax_sc, ax_box) = plt.subplots(
                1, 2, figsize=(11, 5.5), gridspec_kw={'width_ratios': [3.2, 1]},
            )
            for i, level in enumerate(levels):
                sub   = runs[runs[hparam] == level]
                color = (readout_color(level, i) if hparam == 'readout'
                         else plt.cm.tab10(i % 10))
                label = readout_label(level) if hparam == 'readout' else str(level)
                # Multiplicative jitter: on a log axis this is a constant visual width.
                x = sub['n_params'].to_numpy() * rng.uniform(0.85, 1.15, len(sub))
                ax_sc.scatter(x, sub[test_col], s=22, alpha=0.8, color=color,
                              edgecolors='white', linewidths=0.4, label=label, zorder=3)
                ax_box.boxplot(sub[test_col].to_numpy(), positions=[i], widths=0.55,
                               patch_artist=True, showfliers=False, whis=(5, 95),
                               boxprops=dict(facecolor=color, edgecolor='none'),
                               medianprops=dict(color='white', linewidth=1.2),
                               whiskerprops=dict(color='black', linewidth=1.0),
                               capprops=dict(color='none'))

            ax_sc.set_xscale('log')
            ax_sc.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
            ax_sc.set_xlabel('Number of parameters', fontweight='bold')
            ax_sc.set_ylabel(f"test {METRIC_LABELS.get(metric, metric)}", fontweight='bold')
            ax_sc.grid(which='both', color='0.92', linewidth=0.8)
            ax_sc.set_axisbelow(True)
            for side in ('top', 'right'):
                ax_sc.spines[side].set_visible(False)

            # Both panels share the metric axis so the boxes line up with the cloud.
            ax_box.set_ylim(ax_sc.get_ylim())
            ax_box.set_yticklabels([])
            ax_box.set_ylabel(f"test {METRIC_LABELS.get(metric, metric)}", fontweight='bold')
            ax_box.set_xticks(range(len(levels)))
            ax_box.set_xticklabels([readout_label(l) if hparam == 'readout' else str(l)
                                    for l in levels], rotation=90, fontsize=9)
            ax_box.set_xlim(-0.6, len(levels) - 0.4)
            ax_box.set_xlabel(hlabel, fontweight='bold')
            ax_box.grid(axis='y', color='0.92', linewidth=0.8)
            ax_box.set_axisbelow(True)
            for side in ('top', 'right'):
                ax_box.spines[side].set_visible(False)

            ax_box.legend(*ax_sc.get_legend_handles_labels(), title=hlabel,
                          loc='upper left', bbox_to_anchor=(1.15, 1.0),
                          frameon=False, fontsize=9, title_fontsize=9,
                          markerscale=1.6, handletextpad=0.3)

            fig.suptitle(f"{dataset} | {model_label(model)} — architectures colored by {hlabel}",
                         x=0.02, ha='left', fontsize=13, fontweight='bold')
            fig.tight_layout(rect=(0, 0, 0.88, 0.95))
            path = os.path.join(out_dir, f"{dataset}_{model}_{hparam}_architectures.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  saved {path}")
            
def plot_readout_deltas(df: pd.DataFrame, out_dir: str):
    """Paired readout differences against the mean readout, one panel per dataset.

    plot_sweep draws each readout's error bar separately, so config-to-config and
    seed-to-seed variance sits inside every bar and the bars overlap even when one
    readout is consistently better. Pairing each run against the mean-readout run
    with the same (num_layers, dropout, hidden_dim, seed) cancels both, leaving the
    readout effect on its own. A box clear of zero is a real difference.
    """
    os.makedirs(out_dir, exist_ok=True)
    datasets = sorted(df['dataset'].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.5),
                             squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub       = df[df['dataset'] == dataset]
        metric, _ = _primary_metric(sub['task_type'].iloc[0])
        test_col  = f'test_{metric}'

        # One row per (config, seed), one column per readout. A row only survives the
        # dropna below if both the baseline and the compared readout actually ran,
        # which matters while the sweep is still incomplete.
        wide = sub.pivot_table(index=['num_layers', 'dropout', 'hidden_dim', 'seed'],
                               columns='readout', values=test_col)
        if BASELINE_READOUT not in wide.columns:
            continue
        others = [r for r in ordered(wide.columns, READOUT_ORDER) if r != BASELINE_READOUT]
        deltas = {}
        for readout in others:
            pair = wide[[BASELINE_READOUT, readout]].dropna()
            deltas[readout] = (pair[readout] - pair[BASELINE_READOUT]).to_numpy()

        for i, readout in enumerate(others):
            if len(deltas[readout]) == 0:
                continue
            color = readout_color(readout, i)
            ax.boxplot(deltas[readout], positions=[i], widths=0.55, patch_artist=True,
                       showfliers=False, whis=(5, 95),
                       boxprops=dict(facecolor=color, edgecolor='none'),
                       medianprops=dict(color='white', linewidth=1.2),
                       whiskerprops=dict(color='black', linewidth=1.0),
                       capprops=dict(color='none'))

        # Annotate after all boxes, so the y limits are final. The extra headroom keeps
        # the labels off the whiskers, which otherwise reach the top of the frame.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.18 * (hi - lo))
        top = ax.get_ylim()[1]
        for i, readout in enumerate(others):
            d = deltas[readout]
            if len(d) == 0:
                continue
            # Share of paired runs this readout wins; 50% means no effect at all.
            ax.text(i, top, f"{(d > 0).mean():.0%} win\nmed {np.median(d):+.3f}\nn={len(d)}",
                    ha='center', va='top', fontsize=8)

        ax.axhline(0.0, color='black', linewidth=1.0, zorder=1)
        ax.set_xticks(range(len(others)), [readout_label(r) for r in others], fontsize=9)
        ax.set_xlim(-0.6, len(others) - 0.4)
        ax.set_title(dataset, fontweight='bold')
        ax.set_ylabel(f"test {METRIC_LABELS.get(metric, metric)} − {readout_label(BASELINE_READOUT)}")
        ax.grid(axis='y', color='0.92', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    fig.suptitle("Readout effect, paired by config and seed (above zero beats mean pooling)",
                 x=0.02, ha='left', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, 'readout_deltas.png')
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

    # results/sweep/<DATASET>/<model>/<readout>/L<l>_do<d>_h<h>/*_metrics.json
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

    # ---- Headline: test mean ± std (layers, dropout, hidden_dim) ----
    if selected.empty:
        print(f"\nNo config has {args.min_seeds} valid seeds yet.")
        return
    print()
    for _, r in selected.iterrows():
        flags = (" *best*" if r['best_readout'] else "") + (" [grid edge]" if r['at_grid_edge'] else "")
        print(f"  {r['dataset']:8s} | {r['model']:8s} | {r['readout']:9s} | "
              f"test_{r['metric']} {r['test_mean']:.4f} ± {r['test_std']:.4f} "
              f"(L{r['num_layers']}, do{r['dropout']}, h{r['hidden_dim']}) | n={r['n']}{flags}")

    if args.plot:
        print()
        plot_dir = os.path.join(args.results_dir, 'plots')
        plot_sweep(df, selected, plot_dir, min_seeds=args.min_seeds)
        plot_heatmaps(df, selected, plot_dir, min_seeds=args.min_seeds)
        plot_readout_deltas(df, plot_dir)


if __name__ == '__main__':
    main()