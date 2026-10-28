"""Fold-level summaries of ModeTrace results.

Matched: per (dataset, model, readout, perturbation, p, fold) the epoch with the
best val score on the primary metric is selected and every test metric is read
there. OOD: test metrics of the best clean-val checkpoint. Both add `degradation`
vs the clean run of the same fold (positive = worse) and aggregate over folds.
"""

from pathlib import Path

import pandas as pd

from src.results import HIGHER_IS_BETTER, ci95

PRIMARY = {'binary': 'roc_auc', 'regression': 'rmse'}   # metric names as written by protocols.py
CLEAN_KEYS = ['dataset', 'model', 'readout', 'fold', 'metric']
RUN_KEYS = ['dataset', 'model', 'readout', 'perturbation', 'p', 'fold']


def _read_all(directory) -> pd.DataFrame:
    files = sorted(Path(directory).glob('*.csv'))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _add_degradation(df: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(clean[CLEAN_KEYS + ['value']].rename(columns={'value': 'clean_value'}),
                  on=CLEAN_KEYS, how='left')
    sign = df['metric'].map(lambda m: 1.0 if HIGHER_IS_BETTER.get(m, True) else -1.0)
    df['degradation'] = sign * (df['clean_value'] - df['value'])
    return df


def _aggregate(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    g = df.groupby(keys, dropna=False)
    return g.agg(mean=('value', 'mean'), std=('value', 'std'), n=('value', 'count'),
                 ci95=('value', ci95),
                 degradation_mean=('degradation', 'mean'), degradation_std=('degradation', 'std'),
                 degradation_ci95=('degradation', ci95)).reset_index().round(4)


def summarize_matched(runs_dir) -> tuple:
    """(per-fold table, summary) from <runs_dir>/*.csv, or (None, None) if there are none."""
    df = _read_all(runs_dir)
    if df.empty:
        return None, None

    val = df[(df['split'] == 'val') & (df['metric'] == df['task_type'].map(PRIMARY))].dropna(subset=['value'])
    best = []
    for _, grp in val.groupby(RUN_KEYS):
        higher = HIGHER_IS_BETTER[grp['metric'].iloc[0]]
        best.append(grp.loc[grp['value'].idxmax() if higher else grp['value'].idxmin()])
    best = pd.DataFrame(best)[RUN_KEYS + ['epoch']].rename(columns={'epoch': 'best_epoch'})

    test = df[df['split'] == 'test'].merge(best, on=RUN_KEYS)
    folds = test[test['epoch'] == test['best_epoch']][RUN_KEYS + ['task_type', 'best_epoch', 'metric', 'value']]
    folds = _add_degradation(folds, folds[folds['perturbation'] == 'none'])
    summary = _aggregate(folds, ['dataset', 'model', 'readout', 'task_type', 'perturbation', 'p', 'metric'])
    return folds, summary


def summarize_ood(ood_dir):
    """Summary from <ood_dir>/*.csv, or None if there are none."""
    df = _read_all(ood_dir)
    if df.empty:
        return None
    df = _add_degradation(df, df[df['test_perturbation'] == 'none'])
    return _aggregate(df, ['dataset', 'model', 'readout', 'task_type', 'test_perturbation', 'test_p', 'metric'])
