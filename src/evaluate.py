"""Evaluation utilities: predictions, task metrics, and metric files."""

import json

import numpy as np
import torch


def predict(model, loader, device, task):
    """(y_true, y_pred), each [N, num_tasks]; y_pred = task.predict(model output), NaN = missing label."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            y   = batch.y.float().view(batch.num_graphs, -1)
            out = model(batch).view(batch.num_graphs, -1)

            all_preds.append(task.predict(out).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def evaluate(model, loader, device, task) -> dict:
    """{metric: value} for every metric of `task`; multi-task metrics are the mean over tasks."""
    y_true, y_pred = predict(model, loader, device, task)
    scores = {}
    for name, fn in task.metrics.items():
        per_task = []
        for t in range(y_true.shape[1]):
            valid = ~np.isnan(y_true[:, t])
            per_task.append(fn(y_true[valid, t], y_pred[valid, t]))
        per_task = [v for v in per_task if v == v]
        scores[name] = float(np.mean(per_task)) if per_task else float('nan')
    return scores


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
