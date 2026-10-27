"""Dataset loading for MoleculeNet benchmarks (binary classification and regression)."""

import os

import pandas as pd
import torch
from torch_geometric.datasets import MoleculeNet

# task_type selects the loss and metrics (src/tasks.py); only binary splits are class-stratified.
DATASET_REGISTRY = {
    # Binary classification
    'BBBP':     {'name': 'BBBP',     'task_type': 'binary',     'num_tasks': 1},
    'BACE':     {'name': 'BACE',     'task_type': 'binary',     'num_tasks': 1},
    'CLINTOX':  {'name': 'ClinTox',  'task_type': 'binary',     'num_tasks': 2},
    'HIV':      {'name': 'HIV',      'task_type': 'binary',     'num_tasks': 1},
    # Regression
    'ESOL':     {'name': 'ESOL',     'task_type': 'regression', 'num_tasks': 1},   # log solubility
    'FREESOLV': {'name': 'FreeSolv', 'task_type': 'regression', 'num_tasks': 1},   # hydration free energy
    'LIPO':     {'name': 'Lipo',     'task_type': 'regression', 'num_tasks': 1},   # logD at pH 7.4
}


def load_dataset(name: str, root: str = './data'):
    """Load a MoleculeNet dataset by name.

    Returns:
        dataset: PyG InMemoryDataset
        meta:    dict with num_tasks, task_type
    """
    key = name.upper()
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' not in registry. "
            f"Supported: {list(DATASET_REGISTRY.keys())}"
        )
    meta = DATASET_REGISTRY[key]
    dataset = MoleculeNet(root=root, name=meta['name'])
    return dataset, meta


def get_smiles(dataset) -> list:
    """Return SMILES strings for every molecule in a MoleculeNet dataset.

    PyG MoleculeNet stores SMILES as data.smiles on each graph object.
    Returns None for any graph missing the attribute.
    """
    smiles = []
    for data in dataset:
        smiles.append(getattr(data, 'smiles', None))
    return smiles


def feature_dims(graphs) -> tuple:
    """(num_node_features, num_edge_features) of the first graph; 0 edge features without edge_attr."""
    g = graphs[0]
    ea = getattr(g, 'edge_attr', None)
    return int(g.x.shape[1]), (int(ea.shape[1]) if ea is not None and ea.dim() > 1 else 0)


def load_curated_csv(csv_path: str, label_cols: list):
    """Build one PyG graph per row of a curated CSV, in row order.

    Row i of the CSV becomes graph i, so SPECTRA split indices (which refer
    to rows of the curated CSV) can be used directly.

    Returns:
        graphs: list of PyG Data objects
        meta:   dict with num_tasks, task_type (same shape as load_dataset)
    """
    from rdkit import Chem
    from torch_geometric.utils import from_smiles

    df = pd.read_csv(csv_path)
    missing = [c for c in ['smiles'] + list(label_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")

    graphs = []
    for i, (smi, y) in enumerate(zip(df['smiles'], df[label_cols].values)):
        # Fail loudly: skipping a row would shift every later index and
        # silently misalign the SPECTRA splits.
        if Chem.MolFromSmiles(smi) is None:
            raise ValueError(f"RDKit could not parse row {i} of {csv_path}: {smi}")
        g = from_smiles(smi)  # same featurizer PyG MoleculeNet uses
        g.y = torch.tensor(y, dtype=torch.float).view(1, -1)
        g.smiles = smi
        graphs.append(g)

    meta = {
        'name': os.path.splitext(os.path.basename(csv_path))[0],
        'task_type': 'binary',
        'num_tasks': len(label_cols),
    }
    return graphs, meta
