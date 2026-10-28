"""Summarize ModeTrace results into CSVs (see src/modetrace/summary.py).

Writes to --results_dir: matched_folds.csv, matched_summary.csv, ood_summary.csv.

Usage:
    python scripts/modetrace/summarize.py --results_dir results/modetrace
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modetrace.summary import PRIMARY, summarize_matched, summarize_ood


def _print_primary(summary, title):
    print(f"\n== {title} (primary metric) ==")
    print(summary[summary['metric'].isin(PRIMARY.values())].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description='Summarize ModeTrace results.')
    ap.add_argument('--results_dir', default='./results/modetrace')
    root = Path(ap.parse_args().results_dir)

    folds, summary = summarize_matched(root / 'runs')
    if summary is None:
        print(f"No matched results in {root / 'runs'}")
    else:
        folds.to_csv(root / 'matched_folds.csv', index=False)
        summary.to_csv(root / 'matched_summary.csv', index=False)
        _print_primary(summary, 'Matched protocol')

    ood = summarize_ood(root / 'ood')
    if ood is None:
        print(f"No OOD results in {root / 'ood'}")
    else:
        ood.to_csv(root / 'ood_summary.csv', index=False)
        _print_primary(ood, 'OOD protocol')


if __name__ == '__main__':
    main()
