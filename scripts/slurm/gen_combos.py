"""Generate combos.txt for the SLURM job array.

Run from gnn_readout_bio/:
    python scripts/slurm/gen_combos.py
    # then check combos.txt, then:
    N=$(wc -l < combos.txt)
    sbatch --array=1-$N scripts/slurm/run_grid_array.slurm
"""

import argparse
from itertools import product

DEFAULTS = {
    'datasets': ['BBBP', 'BACE'],
    'models':   ['gcn', 'gin', 'gat', 'graphsage'],
    'readouts': ['mean', 'sum', 'max', 'attention'],
    'seeds':    list(range(10)),   # 0–9
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=DEFAULTS['datasets'])
    p.add_argument('--models',   nargs='+', default=DEFAULTS['models'])
    p.add_argument('--readouts', nargs='+', default=DEFAULTS['readouts'])
    p.add_argument('--seeds',    nargs='+', type=int, default=DEFAULTS['seeds'])
    p.add_argument('--output', default='combos.txt')
    args = p.parse_args()

    combos = list(product(
        [d.upper() for d in args.datasets],
        args.models,
        args.readouts,
        args.seeds,
    ))

    with open(args.output, 'w') as f:
        for dataset, model, readout, seed in combos:
            f.write(f"{dataset}\t{model}\t{readout}\t{seed}\n")

    print(f"Wrote {len(combos)} combinations to {args.output}")
    print(f"Submit with:  sbatch --array=1-{len(combos)} scripts/slurm/run_grid_array.slurm")


if __name__ == '__main__':
    main()
