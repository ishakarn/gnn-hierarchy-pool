"""Scaffold splitting for molecular datasets using Bemis-Murcko scaffolds."""

import os
import json
import random
from collections import defaultdict


def _get_scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _split_scaffold_groups(scaffold_sets, train_frac, val_frac, seed):
    """
    Given a list of scaffold groups (already sorted largest-first),
    assign groups to train / val / test.

    Largest groups → train (novel scaffolds in val/test).
    Holdout groups are shuffled with `seed` before being split val/test.
    """
    n = sum(len(g) for g in scaffold_sets)
    train_cut = int(train_frac * n)
    val_cut   = int(val_frac   * n)

    train_groups, holdout_groups = [], []
    train_size = 0
    for group in scaffold_sets:
        if train_size < train_cut:
            train_groups.append(group)
            train_size += len(group)
        else:
            holdout_groups.append(group)

    random.Random(seed).shuffle(holdout_groups)

    val_idx, test_idx = [], []
    for group in holdout_groups:
        if len(val_idx) < val_cut:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    train_idx = [i for g in train_groups for i in g]
    return train_idx, val_idx, test_idx


def scaffold_split(
    smiles_list: list,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    labels: list = None,
) -> tuple:
    """Scaffold split that guarantees both classes appear in val and test.

    When labels are provided the split is **stratified by class**: each
    class's scaffold groups are split independently by the same fractions
    and then merged.  This guarantees both classes in every split regardless
    of how imbalanced the dataset is or how the negatives cluster in scaffold
    space — which is the root cause of BBBP val/test being all-positive under
    a naive scaffold split.

    Without labels (or single-class data) it falls back to a standard
    scaffold split with a seeded holdout shuffle.

    Args:
        smiles_list: SMILES for every molecule (None → empty scaffold).
        labels:      scalar label per molecule (int or float).  Pass None
                     to use the non-stratified fallback.
        seed:        controls the holdout scaffold shuffle within each class.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-5

    def _lbl(i):
        v = labels[i] if labels is not None else None
        return None if (v is None or v != v) else round(float(v))

    # Build scaffold → index map
    scaffold_to_idxs = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold_to_idxs[_get_scaffold(smi) if smi else ''].append(idx)

    # Determine whether we can stratify
    if labels is not None:
        classes = sorted({_lbl(i) for i in range(len(smiles_list))
                          if _lbl(i) is not None})
    else:
        classes = []

    # ── Stratified split (one class at a time, then merge) ──────────────────
    if len(classes) >= 2:
        all_train, all_val, all_test = [], [], []

        for cls in classes:
            # Scaffold groups that contain only molecules of this class
            cls_scaffold = defaultdict(list)
            for idx in range(len(smiles_list)):
                if _lbl(idx) == cls:
                    smi = smiles_list[idx]
                    cls_scaffold[_get_scaffold(smi) if smi else ''].append(idx)

            cls_sets = sorted(cls_scaffold.values(), key=len, reverse=True)

            # Use a per-class seed offset so val/test compositions differ
            # between classes while remaining deterministic.
            cls_seed = seed + int(cls) * 997

            tr, vl, te = _split_scaffold_groups(
                cls_sets, train_frac, val_frac, cls_seed
            )
            all_train.extend(tr)
            all_val.extend(vl)
            all_test.extend(te)

        return all_train, all_val, all_test

    # ── Fallback: standard scaffold split (no label information) ────────────
    all_sets = sorted(scaffold_to_idxs.values(), key=len, reverse=True)
    return _split_scaffold_groups(all_sets, train_frac, val_frac, seed)


def get_or_create_split(
    dataset,
    split_dir: str,
    dataset_name: str,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple:
    """Load scaffold split from disk if cached, otherwise compute and save."""
    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.join(split_dir, f"{dataset_name}_scaffold_seed{seed}.json")

    if os.path.exists(split_path):
        with open(split_path) as f:
            splits = json.load(f)
        print(f"Loaded existing scaffold split from {split_path}")
        return splits['train'], splits['val'], splits['test']

    from src.data import get_smiles
    smiles_list = get_smiles(dataset)

    labels = []
    for data in dataset:
        y = data.y
        if y is not None:
            v = y.view(-1)[0].item()
            labels.append(v if v == v else None)
        else:
            labels.append(None)

    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        labels=labels,
    )

    # Sanity-check class balance
    def _pos_ratio(idxs):
        vals = [labels[i] for i in idxs if labels[i] is not None]
        return sum(1 for v in vals if round(float(v)) == 1) / max(len(vals), 1)

    print(
        f"  class balance — "
        f"train: {_pos_ratio(train_idx):.2f} pos  "
        f"val: {_pos_ratio(val_idx):.2f} pos  "
        f"test: {_pos_ratio(test_idx):.2f} pos"
    )

    with open(split_path, 'w') as f:
        json.dump({'train': train_idx, 'val': val_idx, 'test': test_idx}, f)
    print(f"Saved scaffold split ({len(train_idx)}/{len(val_idx)}/{len(test_idx)}) to {split_path}")

    return train_idx, val_idx, test_idx
