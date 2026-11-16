"""Submit a ModeTrace sweep described by a YAML config, one SLURM job per
(protocol, dataset, model). Jobs resume where they stopped, so resubmitting
after a preemption is safe. See configs/modetrace/*.yaml for the format.

Usage:
    python scripts/modetrace/submit.py configs/modetrace/regression.yaml --dry-run
    python scripts/modetrace/submit.py configs/modetrace/regression.yaml
"""

import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.utils import load_config

WALLTIME = {'matched': '8:00:00', 'ood': '4:00:00'}


def main():
    ap = argparse.ArgumentParser(description='Submit a ModeTrace sweep from a YAML config.')
    ap.add_argument('config')
    ap.add_argument('--dry-run', action='store_true', help='Print the sbatch commands only.')
    args = ap.parse_args()

    sweep = load_config(args.config)
    shared = ['--readouts', *sweep.get('readouts', ['mean']),
              '--perturbations', *sweep['perturbations'],
              '--p', *map(str, sweep['p_values']),
              '--folds', *map(str, sweep['folds']),
              '--config', args.config,
              '--output_dir', sweep['output_dir']]

    os.makedirs(os.path.join(ROOT, 'logs'), exist_ok=True)
    for protocol in sweep.get('protocols', ['matched']):
        for dataset in sweep['datasets']:
            for model in sweep['models']:
                cmd = ['sbatch', '--parsable', f'--job-name=mt_{protocol}_{dataset.lower()}_{model}',
                       f'--time={WALLTIME[protocol]}', 'scripts/slurm/run_modetrace.slurm', protocol,
                       '--datasets', dataset, '--models', model, *shared]
                if args.dry_run:
                    print(' '.join(cmd))
                else:
                    job = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
                    print(f"{protocol:8s} {dataset:9s} {model:5s} -> job {job}")


if __name__ == '__main__':
    main()
