"""Training loop with early stopping on validation ROC-AUC."""

import os

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.evaluate import evaluate_auc


def _train_epoch(model, loader, optimizer, device) -> float:
    """One pass over the training set. Returns mean BCEWithLogitsLoss."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
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

        # Mask missing labels (NaN) for future multi-task compatibility
        valid = ~torch.isnan(y)
        if valid.sum() == 0:
            continue

        loss = criterion(out[valid], y[valid])
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
) -> dict:
    """Full training run with early stopping.

    Saves the best checkpoint (highest val AUC) to `checkpoint_path`.
    At the end, reloads the best checkpoint and reports train/val/test AUC.

    Returns:
        dict with train_auc, val_auc, test_auc, best_epoch, history.
    """
    train_data = [dataset[i] for i in train_idx]
    val_data   = [dataset[i] for i in val_idx]
    test_data  = [dataset[i] for i in test_idx]

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
    best_val_auc     = -1.0
    best_epoch       = 1
    patience_counter = 0
    history          = []

    # Always save an initial checkpoint so we have a fallback even if val_auc
    # is NaN for every epoch (can happen when a split lands all-one-class in val).
    torch.save(model.state_dict(), checkpoint_path)

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device)
        val_auc    = evaluate_auc(model, val_loader, device)
        is_nan     = (val_auc != val_auc)   # True when val_auc is float('nan')

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_auc': None if is_nan else round(val_auc, 6),
        })

        if not is_nan and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        elif not is_nan:
            # Only count patience when we actually have a valid metric
            patience_counter += 1

        auc_str = "nan" if is_nan else f"{val_auc:.4f}"
        if epoch % 10 == 0 or epoch <= 3:
            print(
                f"  Epoch {epoch:4d} | loss {train_loss:.4f} | val_auc {auc_str}"
                + (" *" if (not is_nan and patience_counter == 0) else "")
            )

        if patience_counter >= patience:
            print(
                f"  Early stopping at epoch {epoch}  "
                f"(best val_auc={best_val_auc:.4f} at epoch {best_epoch})"
            )
            break

    # Reload best weights and compute final metrics
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    train_auc = evaluate_auc(model, train_loader, device)
    val_auc   = evaluate_auc(model, val_loader,   device)
    test_auc  = evaluate_auc(model, test_loader,  device)

    return {
        'train_auc':  train_auc,
        'val_auc':    val_auc,
        'test_auc':   test_auc,
        'best_epoch': best_epoch,
        'history':    history,
    }
