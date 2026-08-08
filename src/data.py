"""Dataset loading for MoleculeNet molecular classification benchmarks."""

from torch_geometric.datasets import MoleculeNet

# Registry makes it easy to add ClinTox, HIV, etc. later.
DATASET_REGISTRY = {
    'BBBP':    {'name': 'BBBP',    'task_type': 'binary', 'num_tasks': 1},
    'BACE':    {'name': 'BACE',    'task_type': 'binary', 'num_tasks': 1},
    'CLINTOX': {'name': 'ClinTox', 'task_type': 'binary', 'num_tasks': 2},
    'HIV':     {'name': 'HIV',     'task_type': 'binary', 'num_tasks': 1},
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
