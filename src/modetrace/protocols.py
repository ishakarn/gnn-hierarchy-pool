"""ModeTrace protocols, built on the benchmark's training step and evaluators.

Matched: perturb train/val/test at rate p (seeds fold*3 + 1/2/3, so global shuffles
never mix splits), train for a fixed number of epochs, log every metric every
epoch; the best-val epoch is picked later (summary.py).
OOD: train clean, keep the best clean-val checkpoint, evaluate it on perturbed
test sets (seed fold*3 + 3, as in matched).
"""

import copy
import csv
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from src.evaluate import evaluate
from src.tasks import stack_labels
from src.train import _train_epoch
from src.modetrace.perturbations import apply_perturbation

# Long-format CSV schemas, one row per (split, epoch, metric). OOD keeps the original ModeTrace columns.
MATCHED_COLUMNS = [
    "dataset", "model", "readout", "task_type", "perturbation", "p", "fold",
    "split", "epoch", "metric", "value",
]
OOD_COLUMNS = [
    "dataset", "model", "readout", "task_type", "fold", "protocol",
    "train_perturbation", "train_p", "val_perturbation", "val_p",
    "test_perturbation", "test_p",
    "split", "epoch", "metric", "value",
    "checkpoint_epoch", "checkpoint_selection_metric",
]

# Metric names as written to CSV ('roc_auc' matches the original ModeTrace files).
CSV_METRIC_NAMES = {'auc': 'roc_auc'}


def csv_metric_name(metric: str) -> str:
    return CSV_METRIC_NAMES.get(metric, metric)


def append_rows(csv_path: Path, rows: list, columns: list) -> None:
    """Append rows to a CSV, writing the header only if the file is new/empty."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def _optimizer(model, cfg):
    return torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0))


def run_matched(model, train_graphs, val_graphs, test_graphs,
                perturbation: str, p: float, fold: int, task, cfg: dict, device) -> list:
    """Matched-perturbation run. Returns rows {split, epoch, metric, value}."""
    if perturbation != "none" and p > 0.0:
        train_graphs = apply_perturbation(train_graphs, perturbation, p, fold * 3 + 1)
        val_graphs   = apply_perturbation(val_graphs,   perturbation, p, fold * 3 + 2)
        test_graphs  = apply_perturbation(test_graphs,  perturbation, p, fold * 3 + 3)

    task.fit(stack_labels(train_graphs))
    bs = cfg['batch_size']
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=bs, shuffle=False)

    model = model.to(device)
    optimizer = _optimizer(model, cfg)
    rows = []

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device, task)
        scores = {}
        for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            scores[split] = evaluate(model, loader, device, task)
            for metric, value in scores[split].items():
                rows.append({"split": split, "epoch": epoch,
                             "metric": csv_metric_name(metric), "value": value})

        if epoch % 10 == 0 or epoch <= 3:
            m = task.primary
            print(f"  Epoch {epoch:4d} | loss {train_loss:.4f} | "
                  f"val_{m} {scores['val'][m]:.4f} | test_{m} {scores['test'][m]:.4f}")

    return rows


def run_ood(model, train_graphs, val_graphs, test_graphs,
            test_perturbations: list, fold: int, task, cfg: dict, device) -> list:
    """Train clean; evaluate the best-val checkpoint on the clean test set, then on each
    (perturbation, p) in test_perturbations. Returns one row per test case and metric."""
    task.fit(stack_labels(train_graphs))
    bs = cfg['batch_size']
    train_loader = DataLoader(train_graphs, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=bs, shuffle=False)

    model = model.to(device)
    optimizer = _optimizer(model, cfg)
    metric = task.primary
    best_val, best_epoch, best_state = None, 1, None

    for epoch in range(1, cfg['epochs'] + 1):
        _train_epoch(model, train_loader, optimizer, device, task)
        val_score = evaluate(model, val_loader, device, task)[metric]
        if task.is_better(val_score, best_val):
            best_val, best_epoch = val_score, epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    best_str = "nan" if best_val is None else f"{best_val:.4f}"
    print(f"  best_epoch={best_epoch}  clean_val_{metric}={best_str}")

    eval_cases = [("none", 0.0)] + [(pert, p) for pert, p in test_perturbations if pert != "none" and p > 0]
    rows = []
    for pert, p in eval_cases:
        graphs = test_graphs if pert == "none" else apply_perturbation(test_graphs, pert, p, fold * 3 + 3)
        scores = evaluate(model, DataLoader(graphs, batch_size=bs, shuffle=False), device, task)
        print(f"  test {pert:<5} p={p:<4}  {metric}={scores[metric]:.4f}")
        for name, value in scores.items():
            rows.append({
                "test_perturbation": pert, "test_p": p,
                "split": "test", "epoch": best_epoch,
                "metric": csv_metric_name(name), "value": value,
                "checkpoint_epoch": best_epoch,
                "checkpoint_selection_metric": f"clean_val_{csv_metric_name(metric)}",
            })
    return rows
