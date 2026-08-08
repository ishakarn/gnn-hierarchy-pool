"""
Visualise grid experiment results.

Usage:
    python scripts/plot_results.py --dataset BACE --results_dir results/
    python scripts/plot_results.py --dataset BACE BBBP --results_dir results/
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── aesthetics ────────────────────────────────────────────────────────────────
<<<<<<< HEAD
MODEL_ORDER   = ['gcn', 'gin', 'gat', 'graphsage', 'gcnp']
READOUT_ORDER = ['mean', 'sum', 'max', 'attention', 'selfattention']
MODEL_LABELS  = {'gcn': 'GCN', 'gin': 'GIN', 'gat': 'GAT', 'graphsage': 'GraphSAGE', 'gcnp': 'GCN-P'}
=======
MODEL_ORDER   = ['gcn', 'gin', 'gat', 'graphsage']
READOUT_ORDER = ['mean', 'sum', 'max', 'attention']
MODEL_LABELS  = {'gcn': 'GCN', 'gin': 'GIN', 'gat': 'GAT', 'graphsage': 'GraphSAGE'}
READOUT_LABELS = {'mean': 'Mean', 'sum': 'Sum', 'max': 'Max', 'attention': 'Gated Attention'}
>>>>>>> 7ef85a9962cf4a689c670b54696ab3dc657c852c
READOUT_COLORS = {
    'mean':      '#4C72B0',
    'sum':       '#55A868',
    'max':       '#C44E52',
    'attention': '#8172B2',
    'selfattention': '#4000B2'
}


def _load(results_dir: str, dataset: str) -> pd.DataFrame:
    path = os.path.join(results_dir, 'grid_results.csv')
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['dataset'] = df['dataset'].str.upper()
    df = df[df['dataset'] == dataset.upper()].copy()
    for col in ['train_auc', 'val_auc', 'test_auc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['model']   = pd.Categorical(df['model'],   categories=MODEL_ORDER,   ordered=True)
    df['readout'] = pd.Categorical(df['readout'], categories=READOUT_ORDER, ordered=True)
    return df.sort_values(['model', 'readout'])


def plot_heatmap(ax, df, metric='test_auc', title=None):
    """Model × readout heatmap of mean AUC."""
    pivot = (
        df.groupby(['model', 'readout'])[metric]
        .mean()
        .unstack('readout')
        .reindex(index=MODEL_ORDER, columns=READOUT_ORDER)
    )

    vmin = max(0.5, np.nanmin(pivot.values) - 0.02)
    vmax = min(1.0, np.nanmax(pivot.values) + 0.02)
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(READOUT_ORDER)))
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([READOUT_LABELS[r] for r in READOUT_ORDER], fontsize=11)
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_xlabel('Readout', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(title or f'Mean {metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')

    # Annotate each cell
    for i, m in enumerate(MODEL_ORDER):
        for j, r in enumerate(READOUT_ORDER):
            val = pivot.loc[m, r]
            txt = f'{val:.3f}' if not np.isnan(val) else 'NaN'
            color = 'black' if 0.4 < (val - vmin) / max(vmax - vmin, 1e-6) < 0.85 else 'white'
            ax.text(j, i, txt, ha='center', va='center', fontsize=10, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='ROC-AUC')


def plot_bars(ax, df, metric='test_auc', title=None):
    """Grouped bar chart: models on x-axis, bars per readout, with std error bars."""
    agg = df.groupby(['model', 'readout'])[metric].agg(['mean', 'std']).reset_index()

    n_models  = len(MODEL_ORDER)
    n_readouts = len(READOUT_ORDER)
    width = 0.18
    offsets = np.linspace(-(n_readouts - 1) / 2, (n_readouts - 1) / 2, n_readouts) * width

    for j, readout in enumerate(READOUT_ORDER):
        sub = agg[agg['readout'] == readout].set_index('model').reindex(MODEL_ORDER)
        x = np.arange(n_models) + offsets[j]
        bars = ax.bar(
            x, sub['mean'], width=width,
            color=READOUT_COLORS[readout],
            label=READOUT_LABELS[readout],
            alpha=0.88,
        )
        ax.errorbar(
            x, sub['mean'], yerr=sub['std'],
            fmt='none', color='#333333', capsize=3, linewidth=1.2,
        )

    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_ylabel('ROC-AUC', fontsize=12)
    ax.set_title(title or f'Mean ± Std {metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')
    ax.legend(title='Readout', fontsize=10, title_fontsize=10, loc='lower right')
    ax.set_ylim(0.45, 1.02)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)


def plot_bars_by_readout(ax, df, metric='test_auc', title=None):
    """Grouped bar chart: readouts on x-axis, bars per model, with std error bars."""
    agg = df.groupby(['model', 'readout'])[metric].agg(['mean', 'std']).reset_index()

    n_models  = len(MODEL_ORDER)
    n_readouts = len(READOUT_ORDER)
    width = 0.18
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for j, model in enumerate(MODEL_ORDER):
        sub = agg[agg['model'] == model].set_index('readout').reindex(READOUT_ORDER)
        x = np.arange(n_readouts) + offsets[j]
        bars = ax.bar(
            x, sub['mean'], width=width,
            color=f'C{j}',
            label=MODEL_LABELS[model],
            alpha=0.88,
        )
        ax.errorbar(
            x, sub['mean'], yerr=sub['std'],
            fmt='none', color='#333333', capsize=3, linewidth=1.2,
        )

    ax.set_xticks(np.arange(n_readouts))
    ax.set_xticklabels([READOUT_LABELS[r] for r in READOUT_ORDER], fontsize=11)
    ax.set_ylabel('ROC-AUC', fontsize=12)
    ax.set_title(title or f'Mean ± Std {metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')
    ax.legend(title='Model', fontsize=10, title_fontsize=10, loc='lower right')
    ax.set_ylim(0.45, 1.02)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)


def plot_gap(ax, df, title=None):
    """Bar chart of train–test generalisation gap per model × readout."""
    agg = df.groupby(['model', 'readout'])[['train_auc', 'test_auc']].mean()
    agg['gap'] = agg['train_auc'] - agg['test_auc']
    agg = agg.reset_index()

    n_models   = len(MODEL_ORDER)
    n_readouts = len(READOUT_ORDER)
    width  = 0.18
    offsets = np.linspace(-(n_readouts - 1) / 2, (n_readouts - 1) / 2, n_readouts) * width

    for j, readout in enumerate(READOUT_ORDER):
        sub = agg[agg['readout'] == readout].set_index('model').reindex(MODEL_ORDER)
        x = np.arange(n_models) + offsets[j]
        ax.bar(x, sub['gap'], width=width,
               color=READOUT_COLORS[readout], label=readout.capitalize(), alpha=0.88)

    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_ylabel('Train AUC − Test AUC', fontsize=12)
    ax.set_title(title or 'Generalisation Gap (Train − Test)', fontsize=13, fontweight='bold')
    ax.legend(title='Readout', fontsize=10, title_fontsize=10, loc='upper left')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', linewidth=0.8)


def plot_gap_heatmap(ax, df, title=None):
    """Heatmap of mean train−test generalisation gap."""
    agg = df.groupby(['model', 'readout'])[['train_auc', 'test_auc']].mean()
    agg['gap'] = agg['train_auc'] - agg['test_auc']
    pivot = agg['gap'].unstack('readout').reindex(
        index=MODEL_ORDER, columns=READOUT_ORDER
    )
    vmax = max(0.15, np.nanmax(pivot.values) + 0.02)
    im = ax.imshow(pivot.values, cmap='OrRd', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(READOUT_ORDER)))
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([READOUT_LABELS[r] for r in READOUT_ORDER], fontsize=11)
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_title(title or 'Generalisation Gap: Train − Test AUC', fontsize=13, fontweight='bold')
    for i, m in enumerate(MODEL_ORDER):
        for j, r in enumerate(READOUT_ORDER):
            v = pivot.loc[m, r]
            txt = f'{v:.3f}' if not np.isnan(v) else 'NaN'
            color = 'white' if v > vmax * 0.6 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=10,
                    color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Gap')


def make_dashboard(dataset: str, results_dir: str, out_dir: str):
    df = _load(results_dir, dataset)
    if df.empty:
        print(f"No rows found for {dataset} in {results_dir}/grid_results.csv")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(f'{dataset} — GNN Architecture × Readout Comparison', fontsize=15, fontweight='bold', y=0.995)

    plot_heatmap(axes[0], df, metric='test_auc', title='Test ROC-AUC (mean over seeds)')
    plot_bars(axes[1], df, metric='test_auc', title='Test ROC-AUC ± Std (grouped by Model)')
    plot_bars_by_readout(axes[2], df, metric='test_auc', title='Test ROC-AUC ± Std (grouped by Readout)')

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{dataset}_dashboard.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets',    nargs='+', default=['BACE'])
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--out_dir',     default='./results/figures')
    args = p.parse_args()

    for dataset in args.datasets:
        make_dashboard(dataset.upper(), args.results_dir, args.out_dir)


if __name__ == '__main__':
    main()
