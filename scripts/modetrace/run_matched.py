"""ModeTrace matched-perturbation runs (see src/modetrace/protocols.py).

Writes <output_dir>/runs/{DATASET}__{model}__{readout}__{perturbation}__p{p}.csv.
Completed folds are skipped. Summarize with scripts/modetrace/summarize.py.

Usage:
    # smoke test
    python scripts/modetrace/run_matched.py --datasets BACE --models gin \\
        --perturbations none GNFS --p 1.0 --folds 0 --epochs 2

    # BBBP, GIN/GINE x mean/attention readouts, all perturbations and p, 10 folds
    python scripts/modetrace/run_matched.py --datasets BBBP --models gin gine \\
        --readouts mean attention --perturbations none WNFS GNFS WEFS GEFS JGFS
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from src.tasks import build_task
from src.utils import set_seed
from src.modetrace import P_VALUES
from src.modetrace.cli import add_common_args, describe, make_model, make_task, p_str, rate, resolve_cfg
from src.modetrace.data import load_fold, task_type_of
from src.modetrace.protocols import MATCHED_COLUMNS, append_rows, csv_metric_name, run_matched


def _fold_done(csv_path, fold: int, metric: str, epochs: int) -> bool:
    """A fold is complete when the primary metric has 3 splits x epochs rows."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    df = pd.read_csv(csv_path)
    return int(((df['fold'] == fold) & (df['metric'] == metric)).sum()) >= 3 * epochs


def main():
    ap = argparse.ArgumentParser(description='ModeTrace matched-perturbation runs.')
    add_common_args(ap)
    ap.add_argument('--p', nargs='+', type=rate, default=list(P_VALUES),
                    help='Perturbation rates (default: 0 0.25 0.5 0.75 1).')
    args = ap.parse_args()

    cfg = resolve_cfg(args)
    print(f"[modetrace/matched] {describe(cfg)}  device={args.device}")

    # 'none' and p = 0 are the same clean run: run it once
    clean = [('none', 0.0)] if 'none' in args.perturbations or 0.0 in args.p else []
    pert_ps = clean + [(pert, pv) for pert in args.perturbations if pert != 'none' for pv in args.p if pv > 0]

    for dataset in args.datasets:
        metric = csv_metric_name(build_task({'task_type': task_type_of(dataset)}).primary)

        for fold in args.folds:
            todo = []
            for model_name in args.models:
                for readout in args.readouts:
                    for pert, pv in pert_ps:
                        run_csv = Path(args.output_dir, 'runs',
                                       f"{dataset}__{model_name}__{readout}__{pert}__p{p_str(pv)}.csv")
                        if _fold_done(run_csv, fold, metric, cfg['epochs']):
                            print(f"[SKIP done] {run_csv.name} fold={fold}")
                        else:
                            todo.append((model_name, readout, pert, pv, run_csv))
            if not todo:
                continue

            train_g, val_g, test_g, meta = load_fold(dataset, fold, data_root=args.data_root)
            print(f"\n{dataset} fold {fold}: train={len(train_g)} val={len(val_g)} test={len(test_g)}  "
                  f"task={meta['task_type']}  node_feats={meta['num_node_features']}  "
                  f"edge_feats={meta['num_edge_features']}")

            for model_name, readout, pert, pv, run_csv in todo:
                print(f"[RUN] {dataset}/{model_name}/{readout}/{pert}/p={pv}/fold={fold}")
                set_seed(fold)
                model = make_model(model_name, readout, meta, cfg)
                task  = make_task(cfg, meta)
                rows  = run_matched(model, train_g, val_g, test_g, pert, pv, fold, task, cfg, args.device)

                ids = {'dataset': dataset, 'model': model_name, 'readout': readout,
                       'task_type': meta['task_type'], 'perturbation': pert, 'p': pv, 'fold': fold}
                append_rows(run_csv, [{**ids, **r} for r in rows], MATCHED_COLUMNS)

    print(f"\n[DONE] per-run CSVs in {Path(args.output_dir, 'runs')}")


if __name__ == '__main__':
    main()
