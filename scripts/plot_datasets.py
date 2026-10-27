"""Cross-dataset figures for the benchmark grid and ModeTrace, all datasets side by
side (<out_dir>/combined/) and per dataset (<out_dir>/individual/<DATASET>/).
Figures are listed in the README; each dataset uses its own metric (AUC or RMSE).

Usage:
    python scripts/plot_datasets.py --results_dir results/base --modetrace_dir results/modetrace
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DATASET_REGISTRY
from src.results import (
    EDGE_AWARE_MODELS, MODEL_ORDER, READOUT_ORDER, REGRESSION_METRICS,
    analysis_metric, ci95, dataset_task_type, higher_is_better, load_grid_results,
    metric_label, model_color, model_label, model_marker, ordered, readout_label,
)
from src.modetrace.summary import summarize_matched, summarize_ood

PERTURBATIONS = ['WNFS', 'GNFS', 'WEFS', 'GEFS', 'JGFS']
EDGE_PERTURBATIONS = {'WEFS', 'GEFS'}          # only edge features are shuffled
PERTURB_KEY = ('WNFS/GNFS: node features shuffled within graph / across split  |  '
               'WEFS/GEFS: same for edge features  |  JGFS: node + edge features across split')
READOUT_STYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
DPI = 150


def _arrow(metric):
    return '↑' if higher_is_better(metric) else '↓'


def _dataset_order(datasets):
    """Registry order (binary first, then regression), unknown datasets last."""
    return ordered(datasets, list(DATASET_REGISTRY))


def _grid(n, ncols, w, h):
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark_table(df, metric_of):
    """Mean / 95% CI of the test metric per dataset × model × readout."""
    rows = []
    for (dataset, model, readout), grp in df.groupby(['dataset', 'model', 'readout']):
        vals = grp[f'test_{metric_of[dataset]}']
        rows.append({'dataset': dataset, 'model': model, 'readout': readout,
                     'metric': metric_of[dataset], 'mean': vals.mean(), 'ci95': ci95(vals),
                     'n': int(vals.notna().sum())})
    return pd.DataFrame(rows)


def panel_benchmark_bars(ax, tab, dataset, models, readouts, legend=True):
    sub = tab[tab['dataset'] == dataset]
    metric = sub['metric'].iloc[0]
    width = 0.8 / len(models)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    for j, model in enumerate(models):
        ms = sub[sub['model'] == model].set_index('readout').reindex(readouts)
        x = np.arange(len(readouts)) + offsets[j]
        ax.bar(x, ms['mean'], width=width, color=model_color(model, j), alpha=0.88,
               label=model_label(model))
        ax.errorbar(x, ms['mean'], yerr=ms['ci95'], fmt='none', color='#333', capsize=2.5, linewidth=1)
    ax.set_xticks(np.arange(len(readouts)))
    ax.set_xticklabels([readout_label(r) for r in readouts], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(f'Test {metric_label(metric)}', fontsize=9)
    ax.set_title(f'{dataset} — {metric_label(metric)} {_arrow(metric)}', fontsize=11, fontweight='bold')
    hi = np.nanmax(sub['mean'] + sub['ci95'].fillna(0))
    lo = np.nanmin(sub['mean'] - sub['ci95'].fillna(0))
    if metric == 'auc':
        ax.set_ylim(max(0.4, lo - 0.05), min(1.0, hi + 0.03))
    elif metric in ('rmse', 'mae'):
        ax.set_ylim(0, hi * 1.12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(fontsize=8, loc='best')


def fig_benchmark_bars(tab, datasets, models, readouts, out_dir):
    fig, axes = _grid(len(datasets), 3, 5.2, 3.8)
    for ax, dataset in zip(axes, datasets):
        panel_benchmark_bars(ax, tab, dataset, models, readouts, legend=False)
    handles = [Patch(color=model_color(m, j), label=model_label(m)) for j, m in enumerate(models)]
    fig.legend(handles=handles, loc='lower center', ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Test performance by readout (mean ± 95% CI over seeds)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, os.path.join(out_dir, 'combined', 'benchmark_bars.png'))

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(6, 4))
        panel_benchmark_bars(ax, tab, dataset, models, readouts)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, 'individual', dataset, 'benchmark_bars.png'))


def fig_benchmark_ranking(tab, datasets, models, readouts, out_dir):
    """Configs × datasets, coloured by within-dataset normalised score, annotated with the metric."""
    configs = [(m, r) for m in models for r in readouts
               if ((tab['model'] == m) & (tab['readout'] == r)).any()]
    vals  = np.full((len(configs), len(datasets)), np.nan)
    norm  = np.full_like(vals, np.nan)
    ranks = np.full_like(vals, np.nan)
    for j, dataset in enumerate(datasets):
        sub = tab[tab['dataset'] == dataset].set_index(['model', 'readout'])
        metric = sub['metric'].iloc[0]
        col = np.array([sub['mean'].get(c, np.nan) for c in configs])
        good = col if higher_is_better(metric) else -col
        vals[:, j] = col
        if np.nanmax(good) > np.nanmin(good):
            norm[:, j] = (good - np.nanmin(good)) / (np.nanmax(good) - np.nanmin(good))
        ranks[:, j] = pd.Series(good).rank(ascending=False).values

    mean_rank = np.nanmean(ranks, axis=1)
    fig, ax = plt.subplots(figsize=(1.35 * len(datasets) + 3.2, 0.42 * len(configs) + 1.8))
    im = ax.imshow(norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    for i in range(len(configs)):
        for j in range(len(datasets)):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f'{vals[i, j]:.3f}', ha='center', va='center', fontsize=8,
                        color='black' if 0.2 < norm[i, j] < 0.85 else 'white')
        ax.text(len(datasets) + 0.1, i, f'{mean_rank[i]:.1f}', ha='left', va='center', fontsize=9,
                fontweight='bold')
    ax.text(len(datasets) + 0.1, -0.8, 'mean\nrank', ha='left', va='bottom', fontsize=8)
    metric_names = {d: tab.loc[tab['dataset'] == d, 'metric'].iloc[0] for d in datasets}
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([f'{d}\n{metric_label(metric_names[d])} {_arrow(metric_names[d])}' for d in datasets],
                       fontsize=9)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([f'{model_label(m)} · {readout_label(r)}' for m, r in configs], fontsize=9)
    ax.set_xlim(-0.5, len(datasets) + 0.6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label('within-dataset score (1 = best config)', fontsize=8)
    ax.set_title('Test performance across datasets (cell text = metric value)\n'
                 'colour spans each dataset\'s min–max, so small gaps can look large — check CIs in benchmark_bars',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, 'combined', 'benchmark_ranking.png'))


# ── ModeTrace ─────────────────────────────────────────────────────────────────

def modetrace_tables(modetrace_dir):
    """Matched and OOD summaries with a common (perturbation, p) naming."""
    _, matched = summarize_matched(os.path.join(modetrace_dir, 'runs'))
    ood = summarize_ood(os.path.join(modetrace_dir, 'ood'))
    if ood is not None:
        ood = ood.rename(columns={'test_perturbation': 'perturbation', 'test_p': 'p'})
    return matched, ood


def _modetrace_metric(metric):
    return 'roc_auc' if metric == 'auc' else metric


def _curve_rows(tab, dataset, metric, model, readout, perturbation):
    """Clean point (p = 0) followed by the perturbed points, sorted by p."""
    sub = tab[(tab['dataset'] == dataset) & (tab['metric'] == metric) &
              (tab['model'] == model) & (tab['readout'] == readout)]
    clean = sub[sub['perturbation'] == 'none']
    pert  = sub[sub['perturbation'] == perturbation]
    return pd.concat([clean.assign(p=0.0), pert]).sort_values('p'), clean


def panel_curves(ax, tab, dataset, metric, perturbation, models, readouts):
    for j, model in enumerate(models):
        for k, readout in enumerate(readouts):
            rows, clean = _curve_rows(tab, dataset, metric, model, readout, perturbation)
            if len(rows) < 2:
                continue
            c = model_color(model, j)
            ls = READOUT_STYLES[k % len(READOUT_STYLES)]
            ax.plot(rows['p'], rows['mean'], color=c, linestyle=ls, marker=model_marker(model),
                    markersize=5, linewidth=1.6)
            ax.fill_between(rows['p'], rows['mean'] - rows['std'].fillna(0),
                            rows['mean'] + rows['std'].fillna(0), color=c, alpha=0.15)
            ax.axhline(clean['mean'].iloc[0], color=c, linestyle=':', linewidth=0.8, alpha=0.6)
    na = perturbation in EDGE_PERTURBATIONS and not any(m in EDGE_AWARE_MODELS for m in models)
    ax.set_title(perturbation + ('  (no edge-aware model)' if na else ''), fontsize=10)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '.25', '.5', '.75', '1'])
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)


def _curve_legend(models, readouts):
    handles = [Line2D([0], [0], color=model_color(m, j), marker=model_marker(m), linewidth=1.6,
                      label=model_label(m)) for j, m in enumerate(models)]
    if len(readouts) > 1:
        handles += [Line2D([0], [0], color='#555', linestyle=READOUT_STYLES[k % len(READOUT_STYLES)],
                           label=readout_label(r)) for k, r in enumerate(readouts)]
    return handles


def _curve_rows_for(tab, datasets, metric_of):
    return [d for d in datasets if (tab['dataset'] == d).any()
            and (tab['metric'] == _modetrace_metric(metric_of[d])).any()]


def fig_curves(tab, name, title, datasets, metric_of, models, readouts, out_dir):
    if tab is None:
        return
    datasets = _curve_rows_for(tab, datasets, metric_of)
    perts = [p for p in PERTURBATIONS if (tab['perturbation'] == p).any()]
    if not datasets or not perts:
        return

    def draw(rows, path, suptitle):
        fig, axes = plt.subplots(len(rows), len(perts), figsize=(2.9 * len(perts), 2.6 * len(rows) + 0.6),
                                 squeeze=False, sharex=True)
        for i, dataset in enumerate(rows):
            metric = _modetrace_metric(metric_of[dataset])
            ys = []
            for j, pert in enumerate(perts):
                panel_curves(axes[i, j], tab, dataset, metric, pert, models, readouts)
                ys.append(axes[i, j].get_ylim())
            lo, hi = min(y[0] for y in ys), max(y[1] for y in ys)
            for ax in axes[i]:
                ax.set_ylim(lo, hi)           # shared y within a dataset
            axes[i, 0].set_ylabel(f'{dataset}\nTest {metric_label(metric)} {_arrow(metric)}', fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel('p (fraction shuffled)', fontsize=9)
        fig.legend(handles=_curve_legend(models, readouts), loc='lower center',
                   ncol=len(models) + (len(readouts) if len(readouts) > 1 else 0),
                   frameon=False, bbox_to_anchor=(0.5, -0.02 - 0.02 / len(rows)))
        fig.text(0.5, -0.06 - 0.03 / len(rows), PERTURB_KEY + '  |  dotted: clean (p = 0) level; band: ± std over folds',
                 ha='center', fontsize=7.5, color='#444', style='italic')
        fig.suptitle(suptitle, fontsize=12, fontweight='bold')
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        _save(fig, path)

    draw(datasets, os.path.join(out_dir, 'combined', f'{name}.png'), title)
    for dataset in datasets:
        draw([dataset], os.path.join(out_dir, 'individual', dataset, f'{name}.png'), f'{title} — {dataset}')


def _relative_drop(tab, dataset, metric, model, readout, perturbation, p=1.0):
    sub = tab[(tab['dataset'] == dataset) & (tab['metric'] == metric) &
              (tab['model'] == model) & (tab['readout'] == readout)]
    clean = sub.loc[sub['perturbation'] == 'none', 'mean']
    row = sub[(sub['perturbation'] == perturbation) & np.isclose(sub['p'], p)]
    if clean.empty or row.empty or clean.iloc[0] == 0:
        return np.nan
    return float(row['degradation_mean'].iloc[0] / abs(clean.iloc[0]))


def fig_drop_heatmap(matched, ood, datasets, metric_of, models, readouts, out_dir):
    tables = [(t, n) for t, n in [(matched, 'Matched'), (ood, 'OOD')] if t is not None]
    if not tables:
        return

    def draw(rows_datasets, path):
        rows = [(d, m, r) for d in rows_datasets for m in models for r in readouts
                if any(((t['dataset'] == d) & (t['model'] == m) & (t['readout'] == r)).any() for t, _ in tables)]
        if not rows:
            return
        fig, axes = plt.subplots(1, len(tables), figsize=(1.0 * len(PERTURBATIONS) * len(tables) + 3.5,
                                                          0.42 * len(rows) + 1.8), squeeze=False)
        for ax, (tab, name) in zip(axes[0], tables):
            mat = np.full((len(rows), len(PERTURBATIONS)), np.nan)
            na = np.zeros_like(mat, dtype=bool)
            for i, (d, m, r) in enumerate(rows):
                metric = _modetrace_metric(metric_of[d])
                for j, pert in enumerate(PERTURBATIONS):
                    if pert in EDGE_PERTURBATIONS and m not in EDGE_AWARE_MODELS:
                        na[i, j] = True
                    else:
                        mat[i, j] = 100 * _relative_drop(tab, d, metric, m, r, pert)
            # Diverging scale centred on 0: red = worse than clean, blue = better
            # Capped at ±100% so large regression drops don't wash out the AUC rows (text stays exact)
            vmax = min(100.0, max(10.0, np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 10.0))
            im = ax.imshow(np.where(na, np.nan, mat), cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            for i in range(len(rows)):
                for j in range(len(PERTURBATIONS)):
                    if na[i, j]:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='#eeeeee',
                                                   hatch='///', edgecolor='#bbbbbb', linewidth=0))
                        ax.text(j, i, 'n/a', ha='center', va='center', fontsize=7, color='#888')
                    elif np.isnan(mat[i, j]):
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='#f7f7f7',
                                                   edgecolor='none'))
                        ax.text(j, i, 'not run', ha='center', va='center', fontsize=6.5, color='#aaa')
                    else:
                        ax.text(j, i, f'{mat[i, j]:+.0f}%', ha='center', va='center', fontsize=8,
                                color='white' if abs(mat[i, j]) > 0.6 * vmax else 'black')
            ax.set_xticks(range(len(PERTURBATIONS)))
            ax.set_xticklabels(PERTURBATIONS, fontsize=9)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f'{d} · {model_label(m)}' + (f' · {readout_label(r)}' if len(readouts) > 1 else '')
                                for d, m, r in rows], fontsize=8.5)
            for i in range(1, len(rows)):          # separate datasets
                if rows[i][0] != rows[i - 1][0]:
                    ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.set_title(f'{name}: relative drop at p = 1', fontsize=10, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03).set_label('% worse than clean (blue = better)',
                                                                       fontsize=8)
        fig.text(0.5, -0.02, 'AUC: (clean − perturbed) / clean;  RMSE: (perturbed − clean) / clean.  '
                 'n/a: model never reads edge features.', ha='center', fontsize=7.5, style='italic', color='#444')
        fig.tight_layout()
        _save(fig, path)

    draw(datasets, os.path.join(out_dir, 'combined', 'modetrace_drop.png'))
    for dataset in datasets:
        draw([dataset], os.path.join(out_dir, 'individual', dataset, 'modetrace_drop.png'))


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Combined and per-dataset figures across datasets.')
    p.add_argument('--results_dir',   default='./results/base', help='Benchmark results (grid_results.csv)')
    p.add_argument('--modetrace_dir', default='./results/modetrace', help='ModeTrace results (runs/, ood/)')
    p.add_argument('--datasets', nargs='+', default=None, type=str.upper, help='Default: all with results')
    p.add_argument('--models',   nargs='+', default=['gin', 'gine'])
    p.add_argument('--readouts', nargs='+', default=None, help='Default: all with results')
    p.add_argument('--regression_metric', default='rmse', choices=list(REGRESSION_METRICS))
    p.add_argument('--out_dir', default='./results/figures')
    args = p.parse_args()

    metric_of = {}
    found = []

    grid_path = os.path.join(args.results_dir, 'grid_results.csv')
    if os.path.exists(grid_path):
        df = load_grid_results(args.results_dir, args.datasets)
        df = df[df['model'].isin(args.models)]
        if args.readouts:
            df = df[df['readout'].isin(args.readouts)]
        datasets = _dataset_order(df['dataset'].unique())
        for d in datasets:
            metric_of[d] = analysis_metric(dataset_task_type(df, d), args.regression_metric)
        if datasets:
            models   = ordered(df['model'], MODEL_ORDER)
            readouts = ordered(df['readout'], READOUT_ORDER)
            print(f"Benchmark: {len(df)} runs, datasets {datasets}, models {models}, readouts {readouts}")
            tab = benchmark_table(df, metric_of)
            fig_benchmark_bars(tab, datasets, models, readouts, args.out_dir)
            fig_benchmark_ranking(tab, datasets, models, readouts, args.out_dir)
            found.append('benchmark')
    else:
        print(f"No {grid_path}; skipping benchmark figures (run scripts/aggregate_results.py first)")

    matched, ood = modetrace_tables(args.modetrace_dir)
    tabs = [t for t in (matched, ood) if t is not None]
    if tabs:
        mt = pd.concat(tabs)
        mt = mt[mt['model'].isin(args.models)]
        if args.datasets:
            mt = mt[mt['dataset'].isin(args.datasets)]
        if args.readouts:
            mt = mt[mt['readout'].isin(args.readouts)]
        datasets = _dataset_order(mt['dataset'].unique())
        for d in datasets:
            metric_of.setdefault(d, analysis_metric(mt.loc[mt['dataset'] == d, 'task_type'].iloc[0],
                                                    args.regression_metric))
        models   = ordered(mt['model'], MODEL_ORDER)
        readouts = ordered(mt['readout'], READOUT_ORDER)
        keep = lambda t: None if t is None else t[t['model'].isin(models) & t['readout'].isin(readouts)]
        matched, ood = keep(matched), keep(ood)
        print(f"ModeTrace: datasets {datasets}, models {models}, readouts {readouts}")
        fig_curves(matched, 'modetrace_matched', 'ModeTrace matched protocol (perturb train/val/test)',
                   datasets, metric_of, models, readouts, args.out_dir)
        fig_curves(ood, 'modetrace_ood', 'ModeTrace OOD protocol (train clean, perturb test)',
                   datasets, metric_of, models, readouts, args.out_dir)
        fig_drop_heatmap(matched, ood, datasets, metric_of, models, readouts, args.out_dir)
        found.append('modetrace')
    else:
        print(f"No ModeTrace results in {args.modetrace_dir}; skipping ModeTrace figures")

    if found:
        print(f"\nFigures in {args.out_dir}/combined/ and {args.out_dir}/individual/<DATASET>/")


if __name__ == '__main__':
    main()
