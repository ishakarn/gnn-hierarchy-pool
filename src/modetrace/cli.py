"""Shared CLI and run setup for scripts/modetrace/run_matched.py and run_ood.py."""

import argparse

import torch

from src.data import DATASET_REGISTRY
from src.models import MODEL_REGISTRY, build_model
from src.readouts import READOUTS
from src.tasks import build_task
from src.utils import load_config
from src.modetrace import DEFAULTS, NUM_FOLDS, PERTURBATIONS


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument('--datasets', nargs='+', required=True, type=str.upper,
                    choices=list(DATASET_REGISTRY))
    ap.add_argument('--models',   nargs='+', required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument('--readouts', nargs='+', default=['mean'], choices=list(READOUTS),
                    help='Graph readouts to evaluate (ModeTrace original: mean).')
    ap.add_argument('--perturbations', nargs='+', required=True, choices=list(PERTURBATIONS))
    ap.add_argument('--folds', nargs='+', type=int, default=list(range(NUM_FOLDS)),
                    help=f'Folds to run (default: all {NUM_FOLDS}).')
    ap.add_argument('--config', default=None,
                    help='YAML whose keys override ModeTrace DEFAULTS '
                         '(e.g. configs/bbbp.yaml to use benchmark hyperparameters).')
    ap.add_argument('--epochs', type=int, default=None, help='Override epochs (smoke tests).')
    ap.add_argument('--data_root',  default='./data')
    ap.add_argument('--output_dir', default='./results/modetrace')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')


def rate(value: str) -> float:
    """argparse type for a perturbation rate in [0, 1]."""
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise argparse.ArgumentTypeError(f"perturbation rate must be in [0, 1], got {p}")
    return p


def resolve_cfg(args) -> dict:
    """DEFAULTS <- --config YAML <- CLI overrides."""
    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(load_config(args.config))
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    return cfg


def make_task(cfg: dict, meta: dict):
    return build_task({**cfg, 'task_type': meta['task_type']})


def make_model(model_name: str, readout: str, meta: dict, cfg: dict):
    return build_model(
        model_name, meta['num_node_features'], {**cfg, 'readout': readout},
        num_tasks=meta['num_tasks'], num_edge_features=meta['num_edge_features'],
    )


def p_str(p: float) -> str:
    """0.5 -> '0.5', 1.0 -> '1', 0.0 -> '0' (matches original run-file names)."""
    return str(float(p)).rstrip('0').rstrip('.') or '0'


def describe(cfg: dict) -> str:
    keys = ('hidden_dim', 'num_layers', 'dropout', 'lr', 'weight_decay', 'epochs', 'batch_size', 'pos_weight')
    return '  '.join(f'{k}={cfg[k]}' for k in keys if k in cfg)

