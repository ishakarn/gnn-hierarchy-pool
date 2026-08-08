"""
Statistical analysis of grid experiment results.

Reads grid_results.csv and produces:
  - Summary table: mean/std/CI of test AUC per dataset/model/readout
  - Paired readout comparisons within each dataset/model (same seeds)
  - t-test, Wilcoxon signed-rank, BH FDR correction, Cohen's dz
  - Plots: heatmaps, bars with CI, gen gap, readout rankings, p-value heatmaps

Usage:
    python scripts/analyze_results.py --datasets BACE
    python scripts/analyze_results.py --datasets BACE BBBP
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


# ── constants ─────────────────────────────────────────────────────────────────

MODEL_ORDER    = ['gcn', 'gin', 'gat', 'graphsage']
READOUT_ORDER  = ['mean', 'sum', 'max', 'attention']
MODEL_LABELS   = {'gcn': 'GCN', 'gin': 'GIN', 'gat': 'GAT', 'graphsage': 'GraphSAGE'}
READOUT_COLORS = {
    'mean':      '#4C72B0',
    'sum':       '#55A868',
    'max':       '#C44E52',
    'attention': '#8172B2',
}


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


def _ci95(values):
    """95% CI half-width using t-distribution."""
    n  = len(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    t  = stats.t.ppf(0.975, df=n - 1)
    return float(t * se)


# ── data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir, datasets):
    path = os.path.join(results_dir, 'grid_results.csv')
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['dataset'] = df['dataset'].str.upper()
    if datasets:
        df = df[df['dataset'].isin([d.upper() for d in datasets])].copy()
    for col in ['train_auc', 'val_auc', 'test_auc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['model']   = pd.Categorical(df['model'],   categories=MODEL_ORDER,   ordered=True)
    df['readout'] = pd.Categorical(df['readout'], categories=READOUT_ORDER, ordered=True)
    return df.sort_values(['dataset', 'model', 'readout', 'seed'])


# ── summary table ─────────────────────────────────────────────────────────────

def compute_summary(df):
    """Per dataset/model/readout: mean, std, CI95, gen_gap for all AUC metrics."""
    rows = []
    for (dataset, model, readout), grp in df.groupby(['dataset', 'model', 'readout']):
        for metric in ['train_auc', 'val_auc', 'test_auc']:
            vals = grp[metric].dropna().values
            if len(vals) == 0:
                continue
        row = {'dataset': dataset, 'model': model, 'readout': readout,
               'n_seeds': len(grp)}
        for metric in ['train_auc', 'val_auc', 'test_auc']:
            vals = grp[metric].dropna().values
            if len(vals) > 0:
                row[f'{metric}_mean'] = round(float(np.mean(vals)), 4)
                row[f'{metric}_std']  = round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0), 4)
                row[f'{metric}_ci95'] = round(_ci95(vals) if len(vals) > 1 else float('nan'), 4)
            else:
                row[f'{metric}_mean'] = float('nan')
                row[f'{metric}_std']  = float('nan')
                row[f'{metric}_ci95'] = float('nan')
        train_m = grp['train_auc'].dropna().values
        test_m  = grp['test_auc'].dropna().values
        row['gen_gap_mean'] = round(float(np.mean(train_m) - np.mean(test_m))
                                    if len(train_m) and len(test_m) else float('nan'), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ── paired comparisons ────────────────────────────────────────────────────────

def compute_paired_tests(df):
    """
    For every (dataset, model) pair, compare all C(4,2)=6 readout pairs
    using test_auc values that share the same seed.

    Returns DataFrame with one row per comparison including:
    t-stat, t-test p, Wilcoxon p, BH-adjusted p (within dataset/model block),
    Cohen's dz, mean difference.
    """
    rows = []
    for (dataset, model), grp in df.groupby(['dataset', 'model']):
        block_rows = []
        for rA, rB in itertools.combinations(READOUT_ORDER, 2):
            subA = grp[grp['readout'] == rA].set_index('seed')['test_auc'].dropna()
            subB = grp[grp['readout'] == rB].set_index('seed')['test_auc'].dropna()
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

            block_rows.append({
                'dataset': dataset, 'model': model,
                'readout_A': rA, 'readout_B': rB,
                'n_seeds': len(common_seeds),
                'mean_A': round(float(np.mean(a)), 4),
                'mean_B': round(float(np.mean(b)), 4),
                'mean_diff': round(float(np.mean(d)), 4),   # A - B
                'std_diff':  round(float(np.std(d, ddof=1)), 4),
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

def _heatmap(ax, pivot, title, vmin=None, vmax=None, fmt='.3f', cmap='RdYlGn'):
    vals = pivot.values.astype(float)
    if vmin is None: vmin = max(0.5, np.nanmin(vals) - 0.02)
    if vmax is None: vmax = min(1.0, np.nanmax(vals) + 0.02)
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([r.capitalize() for r in pivot.columns], fontsize=10)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in pivot.index], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = vals[i, j]
            txt = f'{v:{fmt}}' if not np.isnan(v) else 'NaN'
            rng = (vmax - vmin) if vmax != vmin else 1
            rel = (v - vmin) / rng
            color = 'black' if 0.3 < rel < 0.85 else 'white'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_heatmaps(summary, dataset, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'{dataset} — Mean ROC-AUC Heatmaps', fontsize=14, fontweight='bold')

    sub = summary[summary['dataset'] == dataset]
    for ax, metric, title in [
        (axes[0], 'test_auc_mean',  'Test AUC (mean over seeds)'),
        (axes[1], 'val_auc_mean',   'Val AUC (mean over seeds)'),
    ]:
        pivot = sub.pivot(index='model', columns='readout', values=metric) \
                   .reindex(index=MODEL_ORDER, columns=READOUT_ORDER)
        _heatmap(ax, pivot, title)
        ax.set_xlabel('Readout', fontsize=11)
        ax.set_ylabel('Model', fontsize=11)

    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_heatmaps.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_bars_and_gap(summary, dataset, out_dir):
    sub  = summary[summary['dataset'] == dataset]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{dataset} — Test AUC & Generalisation Gap', fontsize=14, fontweight='bold')

    # Bar chart with 95% CI
    ax = axes[0]
    n_m, n_r = len(MODEL_ORDER), len(READOUT_ORDER)
    width = 0.18
    offsets = np.linspace(-(n_r - 1) / 2, (n_r - 1) / 2, n_r) * width
    for j, readout in enumerate(READOUT_ORDER):
        rs = sub[sub['readout'] == readout].set_index('model').reindex(MODEL_ORDER)
        x  = np.arange(n_m) + offsets[j]
        ax.bar(x, rs['test_auc_mean'], width=width,
               color=READOUT_COLORS[readout], label=readout.capitalize(), alpha=0.88)
        ax.errorbar(x, rs['test_auc_mean'], yerr=rs['test_auc_ci95'],
                    fmt='none', color='#333', capsize=3, linewidth=1.2)
    ax.set_xticks(np.arange(n_m))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_ylabel('Test ROC-AUC', fontsize=11)
    ax.set_title('Test AUC ± 95% CI', fontsize=12, fontweight='bold')
    ax.legend(title='Readout', fontsize=9, title_fontsize=9, loc='lower right')
    ax.set_ylim(0.45, 1.02)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # Gen gap
    ax = axes[1]
    for j, readout in enumerate(READOUT_ORDER):
        rs = sub[sub['readout'] == readout].set_index('model').reindex(MODEL_ORDER)
        x  = np.arange(n_m) + offsets[j]
        ax.bar(x, rs['gen_gap_mean'], width=width,
               color=READOUT_COLORS[readout], label=readout.capitalize(), alpha=0.88)
    ax.set_xticks(np.arange(n_m))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_ylabel('Train AUC − Test AUC', fontsize=11)
    ax.set_title('Generalisation Gap', fontsize=12, fontweight='bold')
    ax.legend(title='Readout', fontsize=9, title_fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_bars_gap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_readout_rankings(summary, dataset, out_dir):
    """For each model: horizontal bar showing readout ranking by mean test AUC."""
    sub = summary[summary['dataset'] == dataset]
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(18, 5), sharey=False)
    fig.suptitle(f'{dataset} — Readout Rankings per Model (Test AUC)',
                 fontsize=14, fontweight='bold')

    for ax, model in zip(axes, MODEL_ORDER):
        ms = sub[sub['model'] == model].set_index('readout')
        ranked = ms['test_auc_mean'].reindex(READOUT_ORDER).sort_values(ascending=True)
        ci     = ms['test_auc_ci95'].reindex(ranked.index)
        colors = [READOUT_COLORS[r] for r in ranked.index]
        bars   = ax.barh(range(len(ranked)), ranked.values,
                         color=colors, alpha=0.88, height=0.6)
        ax.errorbar(ranked.values, range(len(ranked)), xerr=ci.values,
                    fmt='none', color='#333', capsize=3, linewidth=1.2)
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels([r.capitalize() for r in ranked.index], fontsize=10)
        ax.set_xlabel('Test AUC', fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight='bold')
        vmin = max(0.4, ranked.min() - 0.05)
        ax.set_xlim(vmin, min(1.0, ranked.max() + 0.05))
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        # Annotate values
        for i, (v, r) in enumerate(zip(ranked.values, ranked.index)):
            if not np.isnan(v):
                ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_readout_rankings.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pvalue_heatmap(paired, dataset, out_dir):
    """4-panel heatmap (one per model): readout × readout BH-adjusted p-values."""
    sub = paired[paired['dataset'] == dataset]
    if sub.empty:
        return

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(20, 5))
    fig.suptitle(f'{dataset} — Pairwise Readout Comparison\n'
                 f'BH-adjusted t-test p-values (test AUC, same seeds)',
                 fontsize=13, fontweight='bold')

    for ax, model in zip(axes, MODEL_ORDER):
        ms = sub[sub['model'] == model]
        mat = np.full((len(READOUT_ORDER), len(READOUT_ORDER)), np.nan)
        ri  = {r: i for i, r in enumerate(READOUT_ORDER)}
        for _, row in ms.iterrows():
            i, j = ri[row['readout_A']], ri[row['readout_B']]
            mat[i, j] = row['t_pval_bh']
            mat[j, i] = row['t_pval_bh']
        np.fill_diagonal(mat, 1.0)

        # Plot -log10(p) for visibility
        log_mat = -np.log10(np.where(mat > 0, mat, 1e-10))
        np.fill_diagonal(log_mat, 0.0)

        im = ax.imshow(log_mat, cmap='YlOrRd', vmin=0, vmax=3, aspect='auto')
        ax.set_xticks(range(len(READOUT_ORDER)))
        ax.set_yticks(range(len(READOUT_ORDER)))
        ax.set_xticklabels([r.capitalize() for r in READOUT_ORDER],
                           rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels([r.capitalize() for r in READOUT_ORDER], fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight='bold')

        for i in range(len(READOUT_ORDER)):
            for j in range(len(READOUT_ORDER)):
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
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--out_dir',     default='./results/analysis')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_results(args.results_dir, args.datasets)
    datasets = sorted(df['dataset'].unique())
    print(f"Loaded {len(df)} runs across datasets: {datasets}")

    # ---- Summary table ----
    summary = compute_summary(df)
    summary_path = os.path.join(args.out_dir, 'summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary → {summary_path}")
    print(summary.to_string(index=False))

    # ---- Paired tests ----
    paired = compute_paired_tests(df)
    if not paired.empty:
        paired_path = os.path.join(args.out_dir, 'paired_comparisons.csv')
        paired.to_csv(paired_path, index=False)
        print(f"\nPaired comparisons → {paired_path}")

        sig = paired[paired['significant_bh'] == True]
        print(f"\nSignificant pairs (BH-corrected t-test p < 0.05): {len(sig)} / {len(paired)}")
        if not sig.empty:
            print(sig[['dataset', 'model', 'readout_A', 'readout_B',
                        'mean_diff', 'cohen_dz', 't_pval_bh']].to_string(index=False))

    # ---- Plots per dataset ----
    for dataset in datasets:
        print(f"\nPlotting {dataset}...")
        plot_heatmaps(summary, dataset, args.out_dir)
        plot_bars_and_gap(summary, dataset, args.out_dir)
        plot_readout_rankings(summary, dataset, args.out_dir)
        if not paired.empty:
            plot_pvalue_heatmap(paired, dataset, args.out_dir)

    print(f"\nAll outputs in {args.out_dir}/")


if __name__ == '__main__':
    main()
