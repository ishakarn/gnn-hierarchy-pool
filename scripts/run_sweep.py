"""Hyperparameter sweep for one (dataset, model, readout): layers x dropout grid, every config on every seed."""

import os
import sys
import argparse
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_grid import DATASET_TO_CONFIG, _run_name, _run_single


# Grid from docs/hparam_sweep_protocol.ipynb
LAYERS   = [2, 3, 4, 5, 6]
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS    = [0, 1, 2, 3, 4]


def run_dir(output_dir, dataset, model, readout, layers, dropout):
    """Folder for one config; holds one metrics file per seed, e.g. BBBP/gin/sum/L4_do0.2."""
    return os.path.join(output_dir, dataset, model, readout, f"L{layers}_do{dropout}")


def main():
    p = argparse.ArgumentParser(
        description="Run the layers x dropout sweep for one dataset/model/readout (resumable)."
    )
    p.add_argument('--dataset',  required=True, choices=list(DATASET_TO_CONFIG.keys()))
    p.add_argument('--model',    required=True, choices=['gin', 'gine', 'chemprop'])
    p.add_argument('--readout',  required=True,
                   choices=['max', 'mean', 'sum', 'attention', 'multihead'])
    p.add_argument('--layers',   nargs='+', type=int,   default=LAYERS)
    p.add_argument('--dropouts', nargs='+', type=float, default=DROPOUTS)
    p.add_argument('--seeds',    nargs='+', type=int,   default=SEEDS)
    p.add_argument('--epochs',   type=int, default=300)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--output_dir', default='./results/sweep')
    args = p.parse_args()

    dataset = args.dataset.upper()
    combos  = list(product(args.layers, args.dropouts, args.seeds))
    print(f"Sweep: {dataset}/{args.model}/{args.readout} | {len(args.layers)} layers × "
          f"{len(args.dropouts)} dropouts × {len(args.seeds)} seeds = {len(combos)} runs\n")

    failed, n_skipped = [], 0
    for i, (layers, dropout, seed) in enumerate(combos, 1):
        out_dir  = run_dir(args.output_dir, dataset, args.model, args.readout, layers, dropout)
        run_name = _run_name(dataset, args.model, args.readout, seed)
        label    = f"L{layers}_do{dropout}/{run_name}"

        # Resume after preemption: a finished run leaves its metrics file behind
        if os.path.exists(os.path.join(out_dir, f"{run_name}_metrics.json")):
            n_skipped += 1
            continue

        print(f"\n[{i}/{len(combos)}] {label}")
        extra_args = [
            '--num_layers', str(layers),
            '--dropout',    str(dropout),
            '--epochs',     str(args.epochs),
            '--patience',   str(args.patience),
        ]
        if not _run_single(dataset, args.model, args.readout, seed, out_dir, extra_args):
            failed.append(label)
            continue

        # Selection only needs metrics; ~12k checkpoints would waste storage
        for ext in ('.pt', '.ckpt'):
            ckpt_path = os.path.join(out_dir, f"{run_name}_best{ext}")
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    print(f"\nDone: {len(combos) - n_skipped - len(failed)} run, {n_skipped} skipped (already done), "
          f"{len(failed)} failed")
    for f in failed:
        print(f"  [FAILED] {f}")


if __name__ == '__main__':
    main()
    