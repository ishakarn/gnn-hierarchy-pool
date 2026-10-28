"""ModeTrace k-fold data loading for the MoleculeNet datasets in DATASET_REGISTRY.

Datasets are loaded with src.data.load_dataset (same ./data cache as the
benchmark). Molecules with a missing first-task label are dropped and only the
first task is kept, as in the original ModeTrace. Scaffold 10-fold CV.
"""

import torch

from src.data import DATASET_REGISTRY, load_dataset, feature_dims
from src.splits import scaffold_kfold_split
from src.modetrace import NUM_FOLDS


def task_type_of(name: str) -> str:
    return DATASET_REGISTRY[name.upper()]['task_type']


def load_fold(name: str, fold: int, k: int = NUM_FOLDS, data_root: str = './data') -> tuple:
    """Load one fold of scaffold k-fold CV.

    Returns:
        train_graphs, val_graphs, test_graphs: lists of PyG Data
        meta: dict with task_type, num_tasks, num_node_features, num_edge_features
    """
    dataset, info = load_dataset(name.upper(), root=data_root)
    graphs, smiles = [], []
    for g in dataset:
        y = g.y.view(1, -1)[:, :1].float()     # first task only -> [1, 1]
        if torch.isnan(y).any():
            continue
        g.y = y
        graphs.append(g)
        smiles.append(getattr(g, 'smiles', '') or '')

    nf, ef = feature_dims(graphs)
    meta = {'task_type': info['task_type'], 'num_tasks': 1,
            'num_node_features': nf, 'num_edge_features': ef}
    train_idx, val_idx, test_idx = scaffold_kfold_split(smiles, fold, k)
    return (
        [graphs[i] for i in train_idx],
        [graphs[i] for i in val_idx],
        [graphs[i] for i in test_idx],
        meta,
    )
