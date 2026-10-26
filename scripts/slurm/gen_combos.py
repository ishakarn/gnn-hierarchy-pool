"""Generate combos.txt for the SLURM job array.

Run from gnn_readout_bio/:
    python scripts/slurm/gen_combos.py
    # then check combos.txt, then:
    N=$(wc -l < combos.txt)
    sbatch --array=1-$N scripts/slurm/run_grid_array.slurm

SPECTRA splits (adds SP and split-replicate columns):
    python scripts/slurm/gen_combos.py --split spectra \
        --sps 0.00 0.10 0.20 0.30 --split_idxs 0 1 2

With --split spectra, splits that run_single.py would skip (missing file,
test set < min_test_size, single-class val/test) are left out of combos.txt
so no GPU job is spent on them. Pass --no_filter to keep them anyway.
"""

import argparse
import os
import sys
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULTS = {
    'datasets':   ['BBBP', 'BACE'],
    'models':     ['gcn', 'gin', 'gat', 'graphsage'],
    'readouts':   ['mean', 'sum', 'max', 'attention'],
    'seeds':      list(range(10)),   # 0–9
    'sps':        [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    'split_idxs': [0, 1, 2],
}


def _usable_spectra_splits(dataset, sps, split_idxs):
    """Return {(sp, split_idx): skip_reason or None} for one dataset."""
    import pandas as pd
    from src.utils import load_config, resolve_spectra_path
    from src.splits import load_spectra_split, SpectraSplitSkipped

    cfg    = load_config(f"configs/{dataset.lower()}.yaml")
    labels = pd.read_csv(resolve_spectra_path(cfg, 'curated_csv'))[cfg['label_cols']].values
    split_dir = resolve_spectra_path(cfg, 'spectra_split_dir')

    status = {}
    for sp, k in product(sps, split_idxs):
        try:
            load_spectra_split(split_dir, sp, k, labels=labels,
                               min_test_size=cfg.get('min_test_size', 20))
            status[(sp, k)] = None
        except SpectraSplitSkipped as e:
            status[(sp, k)] = str(e)
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=DEFAULTS['datasets'])
    p.add_argument('--models',   nargs='+', default=DEFAULTS['models'])
    p.add_argument('--readouts', nargs='+', default=DEFAULTS['readouts'])
    p.add_argument('--seeds',    nargs='+', type=int, default=DEFAULTS['seeds'])
    p.add_argument('--output', default='combos.txt')

    # ---- SPECTRA splits ----
    p.add_argument('--split', choices=['scaffold', 'spectra'], default='scaffold')
    p.add_argument('--sps', nargs='+', type=float, default=DEFAULTS['sps'])
    p.add_argument('--split_idxs', nargs='+', type=int, default=DEFAULTS['split_idxs'])
    p.add_argument('--no_filter', action='store_true',
                   help='Keep SPECTRA splits that run_single.py would skip')
    args = p.parse_args()

    datasets = [d.upper() for d in args.datasets]

    if args.split == 'scaffold':
        combos = list(product(datasets, args.models, args.readouts, args.seeds))
        lines  = ["\t".join(map(str, c)) for c in combos]
    else:
        lines, dropped = [], []
        for dataset in datasets:
            if args.no_filter:
                usable = {(sp, k): None for sp, k in product(args.sps, args.split_idxs)}
            else:
                usable = _usable_spectra_splits(dataset, args.sps, args.split_idxs)
            for (sp, k), reason in usable.items():
                if reason is not None:
                    dropped.append(f"{dataset} SP={sp:.2f} split={k}: {reason}")
                    continue
                for model, readout, seed in product(args.models, args.readouts, args.seeds):
                    lines.append(f"{dataset}\t{model}\t{readout}\t{seed}\t{sp:.2f}\t{k}")
        if dropped:
            print(f"Left out {len(dropped)} SPECTRA split(s) that would be skipped:")
            for d in dropped:
                print(f"  {d}")

    # newline='\n': LF line endings even when generated on Windows, so the
    # SLURM script's `cut` doesn't pick up a stray '\r'
    with open(args.output, 'w', newline='\n') as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Wrote {len(lines)} combinations to {args.output}")
    print(f"Submit with:  sbatch --array=1-{len(lines)} scripts/slurm/run_grid_array.slurm")


if __name__ == '__main__':
    main()
