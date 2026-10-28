"""ModeTrace OOD runs: train clean, test perturbed (see src/modetrace/protocols.py).

Writes <output_dir>/ood/{DATASET}__{model}__{readout}.csv (one file per combination,
so parallel jobs never share a file). Completed folds are skipped.

Usage:
    # smoke test
    python scripts/modetrace/run_ood.py --datasets BACE --models gin gine \\
        --perturbations GNFS GEFS JGFS --p 1.0 --folds 0 --epochs 2

    python scripts/modetrace/run_ood.py --datasets ESOL --models gin gine \\
        --readouts mean sum --perturbations WNFS GNFS WEFS GEFS JGFS
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from src.utils import set_seed
from src.modetrace.cli import add_common_args, describe, make_model, make_task, rate, resolve_cfg
from src.modetrace.data import load_fold
from src.modetrace.protocols import OOD_COLUMNS, append_rows, run_ood


def _fold_done(csv_path, fold: int, cases: list) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    df = pd.read_csv(csv_path)
    sub = df[df['fold'] == fold]
    return all(((sub['test_perturbation'] == pert) & (sub['test_p'] == pv)).any() for pert, pv in cases)


def main():
    ap = argparse.ArgumentParser(description='ModeTrace OOD runs: train clean, test perturbed.')
    add_common_args(ap)
    ap.add_argument('--p', nargs='+', type=rate, default=[0.25, 0.5, 0.75, 1.0],
                    help='Test perturbation rates, all evaluated from one checkpoint (default: 0.25 0.5 0.75 1).')
    args = ap.parse_args()

    cfg = resolve_cfg(args)
    print(f"[modetrace/ood] {describe(cfg)}  device={args.device}")

    cases = [('none', 0.0)] + [
        (pert, pv) for pert in args.perturbations if pert != 'none' for pv in args.p if pv > 0.0
    ]

    for dataset in args.datasets:
        for fold in args.folds:
            train_g, val_g, test_g, meta = load_fold(dataset, fold, data_root=args.data_root)
            print(f"\n{dataset} fold {fold}: train={len(train_g)} val={len(val_g)} "
                  f"test={len(test_g)}  task={meta['task_type']}")

            for model_name in args.models:
                for readout in args.readouts:
                    csv_path = Path(args.output_dir, 'ood', f"{dataset}__{model_name}__{readout}.csv")
                    if _fold_done(csv_path, fold, cases):
                        print(f"[SKIP done] {csv_path.name} fold={fold}")
                        continue

                    print(f"[RUN] {dataset}/{model_name}/{readout}/fold={fold}")
                    set_seed(fold)
                    model = make_model(model_name, readout, meta, cfg)
                    task  = make_task(cfg, meta)
                    rows  = run_ood(model, train_g, val_g, test_g, cases, fold, task, cfg, args.device)

                    ids = {'dataset': dataset, 'model': model_name, 'readout': readout,
                           'task_type': meta['task_type'], 'fold': fold, 'protocol': 'ood_perturbation',
                           'train_perturbation': 'none', 'train_p': 0.0,
                           'val_perturbation': 'none', 'val_p': 0.0}
                    append_rows(csv_path, [{**ids, **r} for r in rows], OOD_COLUMNS)

    print(f"\n[DONE] results in {Path(args.output_dir, 'ood')}")


if __name__ == '__main__':
    main()
