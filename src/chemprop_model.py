"""Chemprop D-MPNN baseline: data bridge from the PyG datasets, and model builder."""

import os
import warnings

import numpy as np
import torch
from chemprop import nn as cnn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score


def pyg_to_chemprop_datapoints(dataset) -> list:
    """Convert a PyG MoleculeNet dataset into Chemprop MoleculeDatapoints.

    The returned list has the same length and order as `dataset`, so
    datapoints[i] is the same molecule as dataset[i] and the split indices
    in data/splits/*.json mean the same molecules for Chemprop.
    """
    datapoints = []
    for i, data in enumerate(dataset):
        smiles = getattr(data, 'smiles', None)
        if smiles is None:
            raise ValueError(f"PyG graph {i} has no .smiles attribute")

        y = np.array(data.y.view(-1).tolist(), dtype=float)

        datapoints.append(MoleculeDatapoint.from_smi(smiles, y=y))
    return datapoints


def build_chemprop_datasets(dataset, train_idx, val_idx, test_idx) -> tuple:
    """Build Chemprop train/val/test MoleculeDatasets using the repo's split indices.

    The indices come from data/splits/*.json (via get_or_create_split), so each
    Chemprop split holds exactly the molecules the PyG models use.
    """
    points = pyg_to_chemprop_datapoints(dataset)
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    return tuple(
        MoleculeDataset([points[i] for i in idx], featurizer)
        for idx in (train_idx, val_idx, test_idx)
    )


class MaxAggregation(cnn.Aggregation):
    """Element-wise max over each molecule's atom embeddings (Chemprop has no built-in max)."""

    def forward(self, H, batch):
        index = batch.unsqueeze(1).repeat(1, H.shape[1])   # [num_atoms, d_h]
        num_graphs = batch.max().int() + 1
        return torch.zeros(num_graphs, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index, H, reduce='amax', include_self=False
        )


def build_chemprop(cfg: dict, readout: str, num_tasks: int = 1) -> MPNN:
    """Return a Chemprop MPNN sized by the same config keys the PyG models use."""
    d_h = cfg['hidden_dim']
    mp = cnn.BondMessagePassing(d_h=d_h, depth=cfg['num_layers'], dropout=cfg['dropout'])

    if readout == 'mean':
        agg = cnn.MeanAggregation()
    elif readout == 'sum':
        agg = cnn.SumAggregation()
    elif readout == 'max':
        agg = MaxAggregation()
    elif readout == 'attention':
        agg = cnn.AttentiveAggregation(output_size=d_h)
    else:
        raise ValueError(f"Readout '{readout}' not supported for chemprop. Choose from: mean, sum, max, attention")

    ffn = cnn.BinaryClassificationFFN(
        n_tasks=num_tasks, input_dim=mp.output_dim, hidden_dim=d_h, dropout=cfg['dropout']
    )

    lr = cfg['lr']
    return MPNN(mp, agg, ffn, init_lr=lr / 10, max_lr=lr, final_lr=lr / 10)


def _chemprop_auc(trainer, model, loader, dset) -> float:
    """Score the best checkpoint on one split, with the same rules as src/evaluate.py."""
    # ckpt_path='best' makes Lightning reload the best checkpoint before predicting.
    probs  = torch.cat(trainer.predict(model, loader, ckpt_path='best')).numpy()   # [n, num_tasks]
    labels = dset.Y
    valid  = ~np.isnan(labels)   # skip missing labels
    labels, probs = labels[valid], probs[valid]

    if len(np.unique(labels)) < 2:
        warnings.warn(
            "Split contains only one class — ROC-AUC undefined, returning NaN."
        )
        return float('nan')

    return float(roc_auc_score(labels, probs))


def train_chemprop(
    model,
    dataset,
    train_idx: list,
    val_idx: list,
    test_idx: list,
    cfg: dict,
    device,
    checkpoint_path: str,
) -> dict:
    """Chemprop counterpart of src.train.train: same inputs, same early-stopping rule.

    Lightning keeps the checkpoint with the highest val AUC (saved as
    `checkpoint_path` with a .ckpt extension). At the end, the best checkpoint
    is reloaded and train/val/test AUC are scored the same way as src/evaluate.py.

    Returns:
        dict with train_auc, val_auc, test_auc, best_epoch.
    """
    train_dset, val_dset, test_dset = build_chemprop_datasets(dataset, train_idx, val_idx, test_idx)

    bs = cfg['batch_size']
    train_loader = build_dataloader(train_dset, batch_size=bs, shuffle=True)
    val_loader   = build_dataloader(val_dset,   batch_size=bs, shuffle=False)
    test_loader  = build_dataloader(test_dset,  batch_size=bs, shuffle=False)
    train_eval_loader = build_dataloader(train_dset, batch_size=bs, shuffle=False)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.dirname(checkpoint_path),
        filename=os.path.splitext(os.path.basename(checkpoint_path))[0],
        monitor='val/roc',
        mode='max',
        enable_version_counter=False,  
    )
    early_stopping = EarlyStopping(monitor='val/roc', mode='max', patience=cfg.get('patience', 20))

    trainer = Trainer(
        accelerator='gpu' if device.type == 'cuda' else 'cpu', 
        devices=1,
        max_epochs=cfg['epochs'],
        callbacks=[checkpoint, early_stopping],
        logger=False,                  
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_loader, val_loader)

    best_epoch = torch.load(checkpoint.best_model_path, weights_only=False)['epoch'] + 1

    return {
        'train_auc':  _chemprop_auc(trainer, model, train_eval_loader, train_dset),
        'val_auc':    _chemprop_auc(trainer, model, val_loader,   val_dset),
        'test_auc':   _chemprop_auc(trainer, model, test_loader,  test_dset),
        'best_epoch': best_epoch,
    }