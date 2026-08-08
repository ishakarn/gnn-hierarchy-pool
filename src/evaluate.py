"""Evaluation utilities for binary molecular classification."""

import json
import warnings

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def evaluate_auc(model, loader, device) -> float:
    """Compute ROC-AUC on a DataLoader.

    Returns float('nan') with a warning if a split contains only one class
    (e.g. a very small test set), rather than crashing.

    Probabilities are computed via sigmoid on raw logits before scoring.
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)   # [batch_size, num_tasks] or [batch_size, 1]
            y   = batch.y.float()

            # Squeeze single-task dimension so shapes are [batch_size]
            if y.dim() == 2 and y.shape[1] == 1:
                y = y.squeeze(-1)
            if out.dim() == 2 and out.shape[1] == 1:
                out = out.squeeze(-1)

            probs = torch.sigmoid(out)

            # Skip NaN labels (future multi-task datasets may have missing values)
            valid = ~torch.isnan(y)
            all_probs.append(probs[valid].cpu().numpy())
            all_labels.append(y[valid].cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    if len(np.unique(all_labels)) < 2:
        warnings.warn(
            "Split contains only one class — ROC-AUC undefined, returning NaN."
        )
        return float('nan')

    return float(roc_auc_score(all_labels, all_probs))


def save_metrics(metrics: dict, path: str):
    """Persist a metrics dict as JSON."""
    # Convert numpy types to native Python for JSON serialisation
    cleaned = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        cleaned[k] = v
    with open(path, 'w') as f:
        json.dump(cleaned, f, indent=2)
