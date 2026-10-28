"""Task types: loss, prediction transform and metrics for binary and regression datasets.

A task is built per run with build_task(cfg) and fitted on the training labels.
Its metric names become result keys: train_auc, ... (binary) or train_rmse, ... (regression).
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, r2_score


def roc_auc(y_true, y_pred) -> float:
    """ROC-AUC, or NaN with a warning if the split contains only one class."""
    if len(np.unique(y_true)) < 2:
        warnings.warn("Split contains only one class — ROC-AUC undefined, returning NaN.")
        return float('nan')
    return float(roc_auc_score(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan')


class _Task:
    metrics = {}            # name -> fn(y_true, y_pred)
    primary = None          # metric used for early stopping and model selection
    higher_is_better = True

    def is_better(self, new: float, best) -> bool:
        """True if `new` improves on `best` (None = no score yet). NaN never improves."""
        if new != new:
            return False
        return best is None or (new > best if self.higher_is_better else new < best)


class BinaryTask(_Task):
    """BCE on logits; predictions are sigmoid probabilities."""

    metrics = {'auc': roc_auc}
    primary = 'auc'
    higher_is_better = True

    def __init__(self, use_pos_weight: bool = False):
        self.use_pos_weight = use_pos_weight
        self.pos_weight = None

    def fit(self, y_train):
        """With use_pos_weight, set pos_weight = n_neg / n_pos from the training labels."""
        if self.use_pos_weight:
            y = np.asarray(y_train, dtype=float).ravel()
            y = y[~np.isnan(y)]
            n_pos = float(y.sum())
            if n_pos > 0:
                self.pos_weight = torch.tensor([(len(y) - n_pos) / n_pos], dtype=torch.float32)
        return self

    def loss(self, out, y):
        valid = ~torch.isnan(y)
        pos_weight = self.pos_weight.to(out.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(out[valid], y[valid], pos_weight=pos_weight)

    def predict(self, out):
        return torch.sigmoid(out)


class RegressionTask(_Task):
    """MSE on targets standardized with train mean/std; predictions are unscaled."""

    metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}
    primary = 'rmse'
    higher_is_better = False

    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self.mean = torch.zeros(1)
        self.std  = torch.ones(1)

    def fit(self, y_train):
        if self.standardize:
            y = np.asarray(y_train, dtype=float).reshape(len(y_train), -1)
            std = np.nanstd(y, axis=0)
            self.mean = torch.tensor(np.nanmean(y, axis=0), dtype=torch.float32)
            self.std  = torch.tensor(np.where(std > 1e-12, std, 1.0), dtype=torch.float32)
        return self

    def loss(self, out, y):
        valid = ~torch.isnan(y)
        y_scaled = (y - self.mean.to(y.device)) / self.std.to(y.device)
        return F.mse_loss(out[valid], y_scaled[valid])

    def predict(self, out):
        return out * self.std.to(out.device) + self.mean.to(out.device)


def build_task(cfg: dict):
    """cfg keys: task_type ('binary' | 'regression'), pos_weight (binary),
    standardize_targets (regression)."""
    task_type = cfg.get('task_type', 'binary')
    if task_type == 'binary':
        return BinaryTask(use_pos_weight=cfg.get('pos_weight', False))
    if task_type == 'regression':
        return RegressionTask(standardize=cfg.get('standardize_targets', True))
    raise ValueError(f"Unknown task_type '{task_type}'. Choose from: binary, regression")


def stack_labels(graphs) -> np.ndarray:
    """Graph labels as an [N, num_tasks] float array (NaN = missing)."""
    return np.stack([g.y.view(-1).float().numpy() for g in graphs])
