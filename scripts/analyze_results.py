"""
Statistical analysis of grid experiment results (binary and regression datasets).

Reads grid_results.csv and produces, per dataset, on that dataset's metric
(ROC-AUC for binary; RMSE by default for regression, see --regression_metric):
  - Summary table: mean/std/CI of every recorded metric per dataset/model/readout
  - Paired readout comparisons within each dataset/model (same seeds)
  - t-test, Wilcoxon signed-rank, BH FDR correction, Cohen's dz
  - Plots: heatmaps, bars with CI, gen gap, readout rankings, p-value heatmaps

gen_gap is signed so that positive = overfitting (train better than test):
train - test for AUC/R², test - train for RMSE/MAE.

Usage:
    python scripts/analyze_results.py --datasets BACE
    python scripts/analyze_results.py --datasets BACE BBBP ESOL
    python scripts/analyze_results.py --datasets ESOL --regression_metric mae
"""

import os
import sys
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.results import (
    MODEL_ORDER, READOUT_ORDER, REGRESSION_METRICS, SPLITS,
    analysis_metric, ci95, color_range, dataset_task_type, higher_is_better,
    load_grid_results, metric_label, model_label, ordered, readout_color, readout_label,
    set_bar_ylim, task_metrics,
)


# ── statistics helpers ────────────────────────────────────────────────────────

def _bh_correction(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    m    = len(pvals)
    if m == 0:
        return np.array([])
    arr  = np.asarray(pvals, dtype=float)
    idx  = np.argsort(arr)
    sorted_p = arr[idx]
    adj  = sorted_p * m / (np.arange(1, m + 1))
    # Enforce monotonicity from right
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    out = np.empty(m)
    out[idx] = adj
    return out


def _cohen_dz(a, b):
    """Cohen's dz for paired data: mean(d) / std(d)."""
    d = np.array(a) - np.array(b)
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 1e-12 else float('nan')


# ── summary table ─────────────────────────────────────────────────────────────

def compute_summary(df, metric_of):
    """Per dataset/model/readout: mean, std, CI95 of every metric, and gen_gap.

    metric_of: {dataset: analysis metric}; gen_gap uses that metric.
    """
    rows = []
    for (dataset, model, readout), grp in df.groupby(['dataset', 'model', 'readout']):
        task_type = grp['task_type'].iloc[0]
        metric    = metric_of[dataset]
        row = {'dataset': dataset, 'model': model, 'readout': readout,
               'task_type': task_type, 'metric': metric, 'n_seeds': len(grp)}
        for m in task_metrics(task_type):
            for split in SPLITS:
                col  = f'{split}_{m}'
                vals = grp[col].dropna().values if col in grp else np.array([])
                if len(vals) > 0:
                    row[f'{col}_mean'] = round(float(np.mean(vals)), 4)
                    row[f'{col}_std']  = round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0), 4)
                    row[f'{col}_ci95'] = round(ci95(vals), 4)
                else:
                    row[f'{col}_mean'] = float('nan')
                    row[f'{col}_std']  = float('nan')
                    row[f'{col}_ci95'] = float('nan')
        train_m = grp[f'train_{metric}'].dropna().values
        test_m  = grp[f'test_{metric}'].dropna().values
        if len(train_m) and len(test_m):
            gap = float(np.mean(train_m) - np.mean(test_m))
            row['gen_gap_mean'] = round(gap if higher_is_better(metric) else -gap, 4)
        else:
            row['gen_gap_mean'] = float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


# ── paired comparisons ────────────────────────────────────────────────────────

def compute_paired_tests(df, metric_of):
    """
    For every (dataset, model) pair, compare all pairs of readouts present
    using test values of the dataset's metric that share the same seed.

    Returns DataFrame with one row per comparison including:
    t-stat, t-test p, Wilcoxon p, BH-adjusted p (within dataset/model block),
    Cohen's dz, mean difference (A - B), and which readout is better on average.
    """
    rows = []
    for (dataset, model), grp in df.groupby(['dataset', 'model']):
        metric = metric_of[dataset]
        col = f'test_{metric}'
        block_rows = []
        for rA, rB in itertools.combinations(ordered(grp['readout'], READOUT_ORDER), 2):
            subA = grp[grp['readout'] == rA].set_index('seed')[col].dropna()
            subB = grp[grp['readout'] == rB].set_index('seed')[col].dropna()
            common_seeds = subA.index.intersection(subB.index)
            if len(common_seeds) < 3:
                continue
            a = subA.loc[common_seeds].values
            b = subB.loc[common_seeds].values
            d = a - b

            t_stat, t_pval = stats.ttest_rel(a, b)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    _, w_pval = stats.wilcoxon(d)
                except Exception:
                    w_pval = float('nan')

            better = rA if (np.mean(d) > 0) == higher_is_better(metric) else rB
            block_rows.append({
                'dataset': dataset, 'model': model, 'metric': metric,
                'readout_A': rA, 'readout_B': rB,
                'n_seeds': len(common_seeds),
                'mean_A': round(float(np.mean(a)), 4),
                'mean_B': round(float(np.mean(b)), 4),
                'mean_diff': round(float(np.mean(d)), 4),   # A - B
                'std_diff':  round(float(np.std(d, ddof=1)), 4),
                'better':    better if np.mean(d) != 0 else 'tie',
                'cohen_dz':  round(_cohen_dz(a, b), 4),
                't_stat':    round(float(t_stat), 4),
                't_pval':    float(t_pval),
                'w_pval':    float(w_pval),
            })

        # BH correction within this dataset/model block
        if block_rows:
            t_pvals = [r['t_pval'] for r in block_rows]
            w_pvals = [r['w_pval'] for r in block_rows]
            t_adj   = _bh_correction(t_pvals)
            w_adj   = _bh_correction(w_pvals)
            for i, r in enumerate(block_rows):
                r['t_pval_bh']  = round(float(t_adj[i]), 5)
                r['w_pval_bh']  = round(float(w_adj[i]), 5)
                r['significant_bh'] = bool(t_adj[i] < 0.05)
            rows.extend(block_rows)

    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

def _heatmap(ax, pivot, title, metric, fmt='.3f'):
    vals = pivot.values.astype(float)
    vmin, vmax = color_range(vals, metric)
    cmap = 'RdYlGn' if higher_is_better(metric) else 'RdYlGn_r'   # green = better
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([readout_label(r) for r in pivot.columns], fontsize=10)
    ax.set_yticklabels([model_label(m) for m in pivot.index], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = vals[i, j]
            txt = f'{v:{fmt}}' if not np.isnan(v) else 'NaN'
            rng = (vmax - vmin) if vmax != vmin else 1
            rel = (v - vmin) / rng
            if not higher_is_better(metric):
                rel = 1 - rel
            color = 'black' if 0.3 < rel < 0.85 else 'white'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_heatmaps(summary, dataset, metric, out_dir):
    sub = summary[summary['dataset'] == dataset]
    models, readouts = ordered(sub['model'], MODEL_ORDER), ordered(sub['readout'], READOUT_ORDER)
    label = metric_label(metric)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, 0.6 * len(models) + 2)))
    fig.suptitle(f'{dataset} — Mean {label} Heatmaps', fontsize=14, fontweight='bold')
    for ax, split in [(axes[0], 'test'), (axes[1], 'val')]:
        pivot = sub.pivot(index='model', columns='readout', values=f'{split}_{metric}_mean') \
                   .reindex(index=models, columns=readouts)
        _heatmap(ax, pivot, f'{split.capitalize()} {label} (mean over seeds)', metric)
        ax.set_xlabel('Readout', fontsize=11)
        ax.set_ylabel('Model', fontsize=11)

    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_heatmaps.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_bars_and_gap(summary, dataset, metric, out_dir):
    sub  = summary[summary['dataset'] == dataset]
    models, readouts = ordered(sub['model'], MODEL_ORDER), ordered(sub['readout'], READOUT_ORDER)
    label = metric_label(metric)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{dataset} — Test {label} & Generalisation Gap', fontsize=14, fontweight='bold')

    # Bar chart with 95% CI
    ax = axes[0]
    n_m, n_r = len(models), len(readouts)
    width = 0.8 / n_r
    offsets = np.linspace(-(n_r - 1) / 2, (n_r - 1) / 2, n_r) * width
    for j, readout in enumerate(readouts):
        rs = sub[sub['readout'] == readout].set_index('model').reindex(models)
        x  = np.arange(n_m) + offsets[j]
        ax.bar(x, rs[f'test_{metric}_mean'], width=width,
               color=readout_color(readout, j), label=readout_label(readout), alpha=0.88)
        ax.errorbar(x, rs[f'test_{metric}_mean'], yerr=rs[f'test_{metric}_ci95'],
                    fmt='none', color='#333', capsize=3, linewidth=1.2)
    ax.set_xticks(np.arange(n_m))
    ax.set_xticklabels([model_label(m) for m in models], fontsize=11)
    ax.set_ylabel(f'Test {label}', fontsize=11)
    ax.set_title(f'Test {label} ± 95% CI', fontsize=12, fontweight='bold')
    ax.legend(title='Readout', fontsize=9, title_fontsize=9,
              loc='lower right' if metric == 'auc' else 'best')
    set_bar_ylim(ax, metric, sub[f'test_{metric}_mean'].values + sub[f'test_{metric}_ci95'].fillna(0).values)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # Gen gap
    ax = axes[1]
    for j, readout in enumerate(readouts):
        rs = sub[sub['readout'] == readout].set_index('model').reindex(models)
        x  = np.arange(n_m) + offsets[j]
        ax.bar(x, rs['gen_gap_mean'], width=width,
               color=readout_color(readout, j), label=readout_label(readout), alpha=0.88)
    ax.set_xticks(np.arange(n_m))
    ax.set_xticklabels([model_label(m) for m in models], fontsize=11)
    gap_label = 'Train − Test' if higher_is_better(metric) else 'Test − Train'
    ax.set_ylabel(f'{gap_label} {label}', fontsize=11)
    ax.set_title('Generalisation Gap (positive = overfit)', fontsize=12, fontweight='bold')
    ax.legend(title='Readout', fontsize=9, title_fontsize=9, loc='best')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_bars_gap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_readout_rankings(summary, dataset, metric, out_dir):
    """For each model: horizontal bars ranking readouts by mean test metric (best on top)."""
    sub = summary[summary['dataset'] == dataset]
    models, readouts = ordered(sub['model'], MODEL_ORDER), ordered(sub['readout'], READOUT_ORDER)
    label = metric_label(metric)
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models) + 1, 5),
                             sharey=False, squeeze=False)
    fig.suptitle(f'{dataset} — Readout Rankings per Model (Test {label})',
                 fontsize=14, fontweight='bold')

    for ax, model in zip(axes[0], models):
        ms = sub[sub['model'] == model].set_index('readout')
        # barh draws index 0 at the bottom, so sort worst -> best
        ranked = ms[f'test_{metric}_mean'].reindex(readouts).dropna() \
                   .sort_values(ascending=higher_is_better(metric))
        ci     = ms[f'test_{metric}_ci95'].reindex(ranked.index)
        colors = [readout_color(r, i) for i, r in enumerate(ranked.index)]
        ax.barh(range(len(ranked)), ranked.values, color=colors, alpha=0.88, height=0.6)
        ax.errorbar(ranked.values, range(len(ranked)), xerr=ci.values,
                    fmt='none', color='#333', capsize=3, linewidth=1.2)
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels([readout_label(r) for r in ranked.index], fontsize=10)
        ax.set_xlabel(f'Test {label}', fontsize=10)
        ax.set_title(model_label(model), fontsize=12, fontweight='bold')
        if len(ranked):
            if metric == 'auc':
                ax.set_xlim(max(0.4, ranked.min() - 0.05), min(1.0, ranked.max() + 0.05))
            else:
                span = max(ranked.max() - ranked.min(), 0.05 * abs(ranked.max()), 1e-3)
                ax.set_xlim(ranked.min() - 0.5 * span, ranked.max() + 0.5 * span)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        # Annotate values just past the error bar
        x_lo, x_hi = ax.get_xlim()
        for i, (v, c) in enumerate(zip(ranked.values, ci.fillna(0).values)):
            ax.text(v + c + 0.01 * (x_hi - x_lo), i + 0.25, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_readout_rankings.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pvalue_heatmap(paired, dataset, out_dir):
    """One panel per model: readout × readout BH-adjusted p-values."""
    sub = paired[paired['dataset'] == dataset]
    if sub.empty:
        return
    models = ordered(sub['model'], MODEL_ORDER)
    readouts = ordered(set(sub['readout_A']) | set(sub['readout_B']), READOUT_ORDER)
    label = metric_label(sub['metric'].iloc[0])

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models) + 1, 5), squeeze=False)
    fig.suptitle(f'{dataset} — Pairwise Readout Comparison\n'
                 f'BH-adjusted t-test p-values (test {label}, same seeds)',
                 fontsize=13, fontweight='bold')

    for ax, model in zip(axes[0], models):
        ms = sub[sub['model'] == model]
        mat = np.full((len(readouts), len(readouts)), np.nan)
        ri  = {r: i for i, r in enumerate(readouts)}
        for _, row in ms.iterrows():
            i, j = ri[row['readout_A']], ri[row['readout_B']]
            mat[i, j] = row['t_pval_bh']
            mat[j, i] = row['t_pval_bh']
        np.fill_diagonal(mat, 1.0)

        # Plot -log10(p) for visibility
        log_mat = -np.log10(np.where(mat > 0, mat, 1e-10))
        np.fill_diagonal(log_mat, 0.0)

        im = ax.imshow(log_mat, cmap='YlOrRd', vmin=0, vmax=3, aspect='auto')
        ax.set_xticks(range(len(readouts)))
        ax.set_yticks(range(len(readouts)))
        ax.set_xticklabels([readout_label(r) for r in readouts],
                           rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels([readout_label(r) for r in readouts], fontsize=9)
        ax.set_title(model_label(model), fontsize=11, fontweight='bold')

        for i in range(len(readouts)):
            for j in range(len(readouts)):
                v = mat[i, j]
                txt = '—' if i == j else (f'{v:.3f}' if not np.isnan(v) else 'NaN')
                sig = '' if i == j or np.isnan(v) else ('*' if v < 0.05 else '')
                ax.text(j, i, f'{txt}{sig}', ha='center', va='center',
                        fontsize=8, color='black')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='-log10(p_BH)')

    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_pvalue_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Analyse GNN grid experiment results.')
    p.add_argument('--datasets',    nargs='+', default=None,
                   help='Datasets to analyse. Default: all in grid_results.csv')
    p.add_argument('--models',      nargs='+', default=None,
                   help='Models to include. Default: all in grid_results.csv')
    p.add_argument('--regression_metric', default='rmse', choices=list(REGRESSION_METRICS),
                   help='Metric for regression datasets (binary datasets always use ROC-AUC).')
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--out_dir',     default=None,
                   help='Default: <results_dir>/analysis')
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    df = load_grid_results(args.results_dir, args.datasets)
    if args.models:
        df = df[df['model'].isin(args.models)]
    datasets = sorted(df['dataset'].unique())
    metric_of = {d: analysis_metric(dataset_task_type(df, d), args.regression_metric) for d in datasets}
    print(f"Loaded {len(df)} runs across datasets: "
          + ', '.join(f'{d} ({metric_label(metric_of[d])})' for d in datasets))

    # ---- Summary table ----
    summary = compute_summary(df, metric_of)
    summary_path = os.path.join(out_dir, 'summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary → {summary_path}")
    for d in datasets:
        m = metric_of[d]
        cols = ['model', 'readout', 'n_seeds'] + [f'{s}_{m}_{stat}' for s in SPLITS for stat in ('mean', 'std')] + ['gen_gap_mean']
        print(f"\n{d} ({metric_label(m)}, {'higher' if higher_is_better(m) else 'lower'} is better)")
        print(summary[summary['dataset'] == d][cols].to_string(index=False))

    # ---- Paired tests ----
    paired = compute_paired_tests(df, metric_of)
    if not paired.empty:
        paired_path = os.path.join(out_dir, 'paired_comparisons.csv')
        paired.to_csv(paired_path, index=False)
        print(f"\nPaired comparisons → {paired_path}")

        sig = paired[paired['significant_bh'] == True]
        print(f"\nSignificant pairs (BH-corrected t-test p < 0.05): {len(sig)} / {len(paired)}")
        if not sig.empty:
            print(sig[['dataset', 'model', 'metric', 'readout_A', 'readout_B',
                        'mean_diff', 'better', 'cohen_dz', 't_pval_bh']].to_string(index=False))

    # ---- Plots per dataset ----
    for dataset in datasets:
        print(f"\nPlotting {dataset}...")
        plot_heatmaps(summary, dataset, metric_of[dataset], out_dir)
        plot_bars_and_gap(summary, dataset, metric_of[dataset], out_dir)
        plot_readout_rankings(summary, dataset, metric_of[dataset], out_dir)
        if not paired.empty:
            plot_pvalue_heatmap(paired, dataset, out_dir)

    print(f"\nAll outputs in {out_dir}/")


if __name__ == '__main__':
    main()
