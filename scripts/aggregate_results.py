"""Collect *_metrics.json files into grid_results.csv + grid_summary.csv.

Looks in --results_dir and its immediate subdirectories (e.g. results/base/BBBP/),
so per-dataset job outputs can be combined into one table for analyze_results.py.

Usage:
    python scripts/aggregate_results.py --results_dir results/base
"""

import argparse
import glob
import json
import os

import pandas as pd


def grid_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of every {split}_{metric} column per dataset/model/readout."""
    numeric = [c for c in df.columns if c.startswith(('train_', 'val_', 'test_'))]
    df = df.assign(**{c: pd.to_numeric(df[c], errors='coerce') for c in numeric})
    summary = (df.groupby(['dataset', 'model', 'readout'])[numeric]
               .agg(['mean', 'std']).round(4).dropna(axis=1, how='all'))
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--results_dir', default='./results')
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, '*_metrics.json'))
                   + glob.glob(os.path.join(args.results_dir, '*', '*_metrics.json')))
    if not files:
        print(f"No *_metrics.json files under {args.results_dir}")
        return

    rows = []
    for f in files:
        with open(f) as fh:
            rows.append(json.load(fh))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, 'grid_results.csv'), index=False)

    summary = grid_summary(df)
    summary.to_csv(os.path.join(args.results_dir, 'grid_summary.csv'))
    print(f"{len(df)} runs from {len(files)} files → {args.results_dir}/grid_results.csv, grid_summary.csv")
    print(summary.to_string())


if __name__ == '__main__':
    main()
