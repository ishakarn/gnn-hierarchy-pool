"""Shared utilities: seeding, device selection, config I/O."""

import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, overrides: dict) -> dict:
    """Shallow merge: override keys from CLI args (None values ignored)."""
    cfg = dict(base)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_spectra_path(cfg: dict, key: str) -> str:
    """Resolve a path inside the SPECTRA_evaluation repo (e.g. key='curated_csv').

    A relative `spectra_repo` is taken relative to this repo's root, so the
    default `../SPECTRA_evaluation` works from any working directory.
    """
    repo = cfg['spectra_repo']
    if not os.path.isabs(repo):
        repo = os.path.join(REPO_ROOT, repo)
    return os.path.normpath(os.path.join(repo, cfg[key]))
