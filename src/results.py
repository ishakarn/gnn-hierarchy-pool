"""Shared helpers for the analysis and plotting scripts.

grid_results.csv has one row per run with {split}_{metric} columns
(binary: *_auc; regression: *_rmse, *_mae, *_r2).
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from src.data import DATASET_REGISTRY
from src.tasks import build_task

SPLITS = ('train', 'val', 'test')

METRIC_LABELS    = {'auc': 'ROC-AUC', 'roc_auc': 'ROC-AUC', 'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R²'}
HIGHER_IS_BETTER = {'auc': True, 'roc_auc': True, 'r2': True, 'rmse': False, 'mae': False}
REGRESSION_METRICS = ('rmse', 'mae', 'r2')

MODEL_ORDER  = ['gin', 'gine', 'chemprop', 'gcn', 'gat', 'graphsage', 'gcnp']
MODEL_LABELS = {'gcn': 'GCN', 'gin': 'GIN', 'gine': 'GINE', 'gat': 'GAT', 'graphsage': 'GraphSAGE',
                'gcnp': 'GCN-P', 'chemprop': 'Chemprop'}
READOUT_ORDER  = ['mean', 'sum', 'max', 'attention', 'selfattention']
READOUT_LABELS = {'mean': 'Mean', 'sum': 'Sum', 'max': 'Max', 'attention': 'Gated Attention',
                  'selfattention': 'Self-Attention'}
# Line colours/markers for model curves (as in the original ModeTrace figures)
MODEL_COLORS  = {'gin': '#1A8C8C', 'gine': '#CC79A7', 'chemprop': '#E69F00',
                 'gcn': '#0072B2', 'gat': '#D55E00', 'graphsage': '#009E73', 'gcnp': '#56B4E9'}
MODEL_MARKERS = {'gin': 'v', 'gine': 'P', 'chemprop': 's',
                 'gcn': 's', 'gat': '^', 'graphsage': 'D', 'gcnp': 'X'}
# Models whose forward pass reads edge_attr: edge perturbations can only affect these.
EDGE_AWARE_MODELS = {'gine'}

READOUT_COLORS = {
    'mean':          '#4C72B0',
    'sum':           '#55A868',
    'max':           '#C44E52',
    'attention':     '#8172B2',
    'selfattention': '#4000B2',
}


def task_metrics(task_type: str) -> list:
    """All metric names recorded for a task type."""
    return list(build_task({'task_type': task_type}).metrics)


def analysis_metric(task_type: str, regression_metric: str = 'rmse') -> str:
    """Metric used for summaries/tests/plots: AUC for binary, chosen metric for regression."""
    return regression_metric if task_type == 'regression' else build_task({'task_type': task_type}).primary


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def higher_is_better(metric: str) -> bool:
    return HIGHER_IS_BETTER.get(metric, True)


def ordered(values, order: list) -> list:
    """Values present in the data, in canonical order; unknown values appended sorted."""
    values = set(values)
    return [v for v in order if v in values] + sorted(v for v in values if v not in order)


def model_label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def readout_label(r: str) -> str:
    return READOUT_LABELS.get(r, r.capitalize())


def model_color(m: str, i: int = 0) -> str:
    return MODEL_COLORS.get(m, f'C{i}')


def model_marker(m: str) -> str:
    return MODEL_MARKERS.get(m, 'o')


def readout_color(r: str, i: int = 0) -> str:
    return READOUT_COLORS.get(r, f'C{i}')


def ci95(values) -> float:
    """95% CI half-width (t-distribution); NaNs ignored, NaN with fewer than 2 values."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float('nan')
    return float(stats.t.ppf(0.975, df=len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))


def color_range(values, metric: str) -> tuple:
    """Colour limits: AUC keeps a [0.5, 1] window; other metrics use the data range."""
    lo, hi = np.nanmin(values), np.nanmax(values)
    if metric == 'auc':
        return max(0.5, lo - 0.02), min(1.0, hi + 0.02)
    pad = 0.02 * (hi - lo) if hi > lo else 0.01 * max(abs(hi), 1.0)
    return lo - pad, hi + pad


def set_bar_ylim(ax, metric: str, values):
    """y-limits for bar charts: [0.45, 1] for AUC, from 0 for error metrics."""
    if metric == 'auc':
        ax.set_ylim(0.45, 1.02)
    elif metric in ('rmse', 'mae'):
        ax.set_ylim(0, np.nanmax(values) * 1.15)


def load_grid_results(results_dir: str, datasets=None) -> pd.DataFrame:
    """Read <results_dir>/grid_results.csv; fill task_type; numeric metric columns."""
    df = pd.read_csv(os.path.join(results_dir, 'grid_results.csv'))
    df.columns = df.columns.str.strip()
    df['dataset'] = df['dataset'].str.upper()
    if datasets:
        df = df[df['dataset'].isin([d.upper() for d in datasets])].copy()
    if 'status' in df.columns:
        df = df[df['status'] != 'skipped'].copy()      # SPECTRA splits run_single.py skipped
    if 'sp' in df.columns and df['sp'].nunique() > 1:
        raise ValueError(f"{results_dir} mixes SPECTRA parameters {sorted(float(v) for v in df['sp'].dropna().unique())}; "
                         "these scripts would average over them. Analyze one SP at a time.")

    registry_type = df['dataset'].map(lambda d: DATASET_REGISTRY.get(d, {}).get('task_type', 'binary'))
    df['task_type'] = df['task_type'].fillna(registry_type) if 'task_type' in df.columns else registry_type

    for col in df.columns:
        if col.startswith(tuple(f'{s}_' for s in SPLITS)):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def dataset_task_type(df: pd.DataFrame, dataset: str) -> str:
    types = df.loc[df['dataset'] == dataset, 'task_type'].dropna().unique()
    if len(types) > 1:
        raise ValueError(f"{dataset}: mixed task types in results: {list(types)}")
    return types[0] if len(types) else DATASET_REGISTRY.get(dataset, {}).get('task_type', 'binary')
