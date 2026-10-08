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
