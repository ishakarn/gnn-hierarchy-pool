"""Training loop with early stopping on the task's validation metric (AUC or RMSE)."""

import torch
from torch_geometric.loader import DataLoader

from src.evaluate import evaluate
from src.tasks import build_task, stack_labels


def _train_epoch(model, loader, optimizer, device, task) -> float:
    """One pass over the training set. Returns mean loss per valid label."""
    model.train()
    total_loss, total_valid = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)   # [B, num_tasks] or [B, 1]
        y   = batch.y.float()

        # Squeeze single-task dim -> [B]
        if y.dim() == 2 and y.shape[1] == 1:
            y = y.squeeze(-1)
        if out.dim() == 2 and out.shape[1] == 1:
            out = out.squeeze(-1)

        # Missing labels (NaN) are masked in task.loss
        valid = ~torch.isnan(y)
        if valid.sum() == 0:
            continue

        loss = task.loss(out, y)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item() * valid.sum().item()
        total_valid += valid.sum().item()

    return total_loss / max(total_valid, 1)


def train(
    model,
    dataset,
    train_idx: list,
    val_idx: list,
    test_idx: list,
    cfg: dict,
    device,
    checkpoint_path: str,
    task=None,
) -> dict:
    """Train with early stopping on the task's primary val metric, keeping the best
    checkpoint at `checkpoint_path`. `task` defaults to build_task(cfg).

    Returns {split}_{metric} for every split and metric (binary: train_auc,
    val_auc, test_auc), best_epoch and history.
    """
    train_data = [dataset[i] for i in train_idx]
    val_data   = [dataset[i] for i in val_idx]
    test_data  = [dataset[i] for i in test_idx]

    task = task or build_task(cfg)
    task.fit(stack_labels(train_data))
    metric = task.primary

    bs = cfg['batch_size']
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=bs, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=bs, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg.get('weight_decay', 0.0),
    )

    patience         = cfg.get('patience', 20)
    best_val         = None
    best_epoch       = 1
    patience_counter = 0
    history          = []

    # Always save an initial checkpoint so we have a fallback even if the val
    # metric is NaN for every epoch (can happen when val lands all-one-class).
    torch.save(model.state_dict(), checkpoint_path)

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device, task)
        val_score  = evaluate(model, val_loader, device, task)[metric]
        is_nan     = (val_score != val_score)   # True when val_score is float('nan')

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            f'val_{metric}': None if is_nan else round(val_score, 6),
        })

        if task.is_better(val_score, best_val):
            best_val   = val_score
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        elif not is_nan:
            # Only count patience when we actually have a valid metric
            patience_counter += 1

        score_str = "nan" if is_nan else f"{val_score:.4f}"
        if epoch % 10 == 0 or epoch <= 3:
            print(
                f"  Epoch {epoch:4d} | loss {train_loss:.4f} | val_{metric} {score_str}"
                + (" *" if (not is_nan and patience_counter == 0) else "")
            )

        if patience_counter >= patience:
            print(
                f"  Early stopping at epoch {epoch}  "
                f"(best val_{metric}={best_val:.4f} at epoch {best_epoch})"
            )
            break

    # Reload best weights and compute final metrics
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    results = {}
    for split, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        for name, value in evaluate(model, loader, device, task).items():
            results[f'{split}_{name}'] = value

    results.update({
        'best_epoch': best_epoch,
        'history':    history,
    })
    return results
