"""
Visualise grid experiment results (binary and regression datasets).

One dashboard per dataset on that dataset's metric: ROC-AUC for binary,
RMSE by default for regression (--regression_metric).

Usage:
    python scripts/plot_results.py --datasets BACE --results_dir results/
    python scripts/plot_results.py --datasets BACE BBBP ESOL --results_dir results/
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.results import (
    MODEL_ORDER, READOUT_ORDER, REGRESSION_METRICS,
    analysis_metric, color_range, dataset_task_type, higher_is_better, load_grid_results,
    metric_label, model_label, ordered, readout_color, readout_label, set_bar_ylim,
)


def plot_heatmap(ax, df, models, readouts, metric, split='test', title=None):
    """Model × readout heatmap of the mean metric (green = better)."""
    col = f'{split}_{metric}'
    pivot = (
        df.groupby(['model', 'readout'])[col]
        .mean()
        .unstack('readout')
        .reindex(index=models, columns=readouts)
    )

    vmin, vmax = color_range(pivot.values, metric)
    cmap = 'RdYlGn' if higher_is_better(metric) else 'RdYlGn_r'
    im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(readouts)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([readout_label(r) for r in readouts], fontsize=11)
    ax.set_yticklabels([model_label(m) for m in models], fontsize=11)
    ax.set_xlabel('Readout', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(title or f'Mean {split.capitalize()} {metric_label(metric)}', fontsize=13, fontweight='bold')

    # Annotate each cell
    for i, m in enumerate(models):
        for j, r in enumerate(readouts):
            val = pivot.loc[m, r]
            txt = f'{val:.3f}' if not np.isnan(val) else 'NaN'
            rel = (val - vmin) / max(vmax - vmin, 1e-6)
            if not higher_is_better(metric):
                rel = 1 - rel
            color = 'black' if 0.4 < rel < 0.85 else 'white'
            ax.text(j, i, txt, ha='center', va='center', fontsize=10, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric_label(metric))


def plot_bars(ax, df, models, readouts, metric, split='test', title=None):
    """Grouped bar chart: models on x-axis, bars per readout, with std error bars."""
    col = f'{split}_{metric}'
    agg = df.groupby(['model', 'readout'])[col].agg(['mean', 'std']).reset_index()

    n_models   = len(models)
    n_readouts = len(readouts)
    width = 0.8 / n_readouts
    offsets = np.linspace(-(n_readouts - 1) / 2, (n_readouts - 1) / 2, n_readouts) * width

    for j, readout in enumerate(readouts):
        sub = agg[agg['readout'] == readout].set_index('model').reindex(models)
        x = np.arange(n_models) + offsets[j]
        ax.bar(
            x, sub['mean'], width=width,
            color=readout_color(readout, j),
            label=readout_label(readout),
            alpha=0.88,
        )
        ax.errorbar(
            x, sub['mean'], yerr=sub['std'],
            fmt='none', color='#333333', capsize=3, linewidth=1.2,
        )

    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels([model_label(m) for m in models], fontsize=11)
    ax.set_ylabel(metric_label(metric), fontsize=12)
    ax.set_title(title or f'Mean ± Std {split.capitalize()} {metric_label(metric)}', fontsize=13, fontweight='bold')
    ax.legend(title='Readout', fontsize=10, title_fontsize=10,
              loc='lower right' if metric == 'auc' else 'best')
    set_bar_ylim(ax, metric, agg['mean'].values + agg['std'].fillna(0).values)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)


def plot_bars_by_readout(ax, df, models, readouts, metric, split='test', title=None):
    """Grouped bar chart: readouts on x-axis, bars per model, with std error bars."""
    col = f'{split}_{metric}'
    agg = df.groupby(['model', 'readout'])[col].agg(['mean', 'std']).reset_index()

    n_models   = len(models)
    n_readouts = len(readouts)
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for j, model in enumerate(models):
        sub = agg[agg['model'] == model].set_index('readout').reindex(readouts)
        x = np.arange(n_readouts) + offsets[j]
        ax.bar(
            x, sub['mean'], width=width,
            color=f'C{j}',
            label=model_label(model),
            alpha=0.88,
        )
        ax.errorbar(
            x, sub['mean'], yerr=sub['std'],
            fmt='none', color='#333333', capsize=3, linewidth=1.2,
        )

    ax.set_xticks(np.arange(n_readouts))
    ax.set_xticklabels([readout_label(r) for r in readouts], fontsize=11)
    ax.set_ylabel(metric_label(metric), fontsize=12)
    ax.set_title(title or f'Mean ± Std {split.capitalize()} {metric_label(metric)}', fontsize=13, fontweight='bold')
    ax.legend(title='Model', fontsize=10, title_fontsize=10,
              loc='lower right' if metric == 'auc' else 'best')
    set_bar_ylim(ax, metric, agg['mean'].values + agg['std'].fillna(0).values)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)


def _gap(df, metric):
    """Mean generalisation gap per model × readout, positive = overfit."""
    agg = df.groupby(['model', 'readout'])[[f'train_{metric}', f'test_{metric}']].mean()
    gap = agg[f'train_{metric}'] - agg[f'test_{metric}']
    return gap if higher_is_better(metric) else -gap


def plot_gap(ax, df, models, readouts, metric, title=None):
    """Bar chart of the train–test generalisation gap per model × readout."""
    gap = _gap(df, metric).rename('gap').reset_index()

    n_models   = len(models)
    n_readouts = len(readouts)
    width  = 0.8 / n_readouts
    offsets = np.linspace(-(n_readouts - 1) / 2, (n_readouts - 1) / 2, n_readouts) * width

    for j, readout in enumerate(readouts):
        sub = gap[gap['readout'] == readout].set_index('model').reindex(models)
        x = np.arange(n_models) + offsets[j]
        ax.bar(x, sub['gap'], width=width,
               color=readout_color(readout, j), label=readout_label(readout), alpha=0.88)

    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels([model_label(m) for m in models], fontsize=11)
    ax.set_ylabel(f'Generalisation gap ({metric_label(metric)})', fontsize=12)
    ax.set_title(title or 'Generalisation Gap (positive = overfit)', fontsize=13, fontweight='bold')
    ax.legend(title='Readout', fontsize=10, title_fontsize=10, loc='upper left')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', linewidth=0.8)


def plot_gap_heatmap(ax, df, models, readouts, metric, title=None):
    """Heatmap of the mean generalisation gap (positive = overfit)."""
    pivot = _gap(df, metric).unstack('readout').reindex(index=models, columns=readouts)
    vmax = np.nanmax(pivot.values)
    vmax = max(0.15, vmax + 0.02) if metric == 'auc' else max(vmax * 1.1, 1e-6)
    im = ax.imshow(pivot.values, cmap='OrRd', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(readouts)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([readout_label(r) for r in readouts], fontsize=11)
    ax.set_yticklabels([model_label(m) for m in models], fontsize=11)
    ax.set_title(title or f'Generalisation Gap ({metric_label(metric)})', fontsize=13, fontweight='bold')
    for i, m in enumerate(models):
        for j, r in enumerate(readouts):
            v = pivot.loc[m, r]
            txt = f'{v:.3f}' if not np.isnan(v) else 'NaN'
            color = 'white' if v > vmax * 0.6 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=10,
                    color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Gap')


def make_dashboard(df_all, dataset: str, out_dir: str, regression_metric: str = 'rmse'):
    df = df_all[df_all['dataset'] == dataset]
    if df.empty:
        print(f"No rows found for {dataset} in grid_results.csv")
        return

    metric   = analysis_metric(dataset_task_type(df_all, dataset), regression_metric)
    label    = metric_label(metric)
    models   = ordered(df['model'], MODEL_ORDER)
    readouts = ordered(df['readout'], READOUT_ORDER)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(f'{dataset} — GNN Architecture × Readout Comparison ({label})',
                 fontsize=15, fontweight='bold', y=0.995)

    plot_heatmap(axes[0], df, models, readouts, metric, title=f'Test {label} (mean over seeds)')
    plot_bars(axes[1], df, models, readouts, metric, title=f'Test {label} ± Std (grouped by Model)')
    plot_bars_by_readout(axes[2], df, models, readouts, metric, title=f'Test {label} ± Std (grouped by Readout)')

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{dataset}_dashboard.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets',    nargs='+', default=['BACE'])
    p.add_argument('--models',      nargs='+', default=None, help='Default: all in grid_results.csv')
    p.add_argument('--regression_metric', default='rmse', choices=list(REGRESSION_METRICS))
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--out_dir',     default=None, help='Default: <results_dir>/figures')
    args = p.parse_args()

    df = load_grid_results(args.results_dir, args.datasets)
    if args.models:
        df = df[df['model'].isin(args.models)]
    out_dir = args.out_dir or os.path.join(args.results_dir, 'figures')
    for dataset in args.datasets:
        make_dashboard(df, dataset.upper(), out_dir, args.regression_metric)


if __name__ == '__main__':
    main()
