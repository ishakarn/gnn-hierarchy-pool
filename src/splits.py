"""Splits: seeded 80/10/10 Bemis-Murcko scaffold split for the benchmark (cached to disk),
and scaffold 10-fold CV for ModeTrace."""

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
    task_type: str = None,
) -> tuple:
    """Load scaffold split from disk if cached, otherwise compute and save.

    Class-stratified for binary datasets only; task_type defaults to the dataset's registry entry.
    """
    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.join(split_dir, f"{dataset_name}_scaffold_seed{seed}.json")

    if os.path.exists(split_path):
        with open(split_path) as f:
            splits = json.load(f)
        print(f"Loaded existing scaffold split from {split_path}")
        return splits['train'], splits['val'], splits['test']

    from src.data import get_smiles, DATASET_REGISTRY
    smiles_list = get_smiles(dataset)

    if task_type is None:
        task_type = DATASET_REGISTRY.get(dataset_name.upper(), {}).get('task_type', 'binary')
    stratify = task_type == 'binary'

    # Stratify on the label; multi-task (ClinTox) on the joint label pattern sum(label_t * 2**t),
    # so every task keeps both classes.
    labels, task_labels = [], []
    for data in dataset:
        y = data.y
        vals = y.view(-1).tolist() if y is not None else [float('nan')]
        task_labels.append(vals)
        if len(vals) == 1:
            v = vals[0]
            labels.append(v if v == v else None)
        elif any(v != v for v in vals):
            labels.append(None)
        else:
            labels.append(float(sum(int(round(v)) << t for t, v in enumerate(vals))))

    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        labels=labels if stratify else None,
    )

    # Sanity-check class balance (per task)
    def _pos_ratio(idxs, t):
        vals = [task_labels[i][t] for i in idxs if task_labels[i][t] == task_labels[i][t]]
        return sum(1 for v in vals if round(float(v)) == 1) / max(len(vals), 1)

    if stratify:
        num_tasks = len(task_labels[0])
        for t in range(num_tasks):
            print(
                f"  class balance{f' (task {t})' if num_tasks > 1 else ''} — "
                f"train: {_pos_ratio(train_idx, t):.2f} pos  "
                f"val: {_pos_ratio(val_idx, t):.2f} pos  "
                f"test: {_pos_ratio(test_idx, t):.2f} pos"
            )

    with open(split_path, 'w') as f:
        json.dump({'train': train_idx, 'val': val_idx, 'test': test_idx}, f)
    print(f"Saved scaffold split ({len(train_idx)}/{len(val_idx)}/{len(test_idx)}) to {split_path}")

    return train_idx, val_idx, test_idx


# ── scaffold k-fold CV (ModeTrace protocol) ─────────────────────────────────────────────
# fold i = test, fold (i+1) % k = val, the rest = train. val/test are returned
# in set-iteration order, as in the original ModeTrace code: GNFS/GEFS permute
# features over the concatenated split, so list order fixes which permutation
# a given seed produces.

def scaffold_kfold_split(smiles_list: list, fold: int, k: int = 10) -> tuple:
    """Scaffold k-fold CV. Scaffold groups, largest first (ties: lowest index),
    are dealt to folds cyclically. Deterministic, label-free (so it also works
    for regression).
    """
    scaffold_to_idxs = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold_to_idxs[_get_scaffold(smi) if smi else ''].append(idx)
    scaffold_sets = sorted(scaffold_to_idxs.values(), key=lambda g: (-len(g), min(g)))

    buckets = [[] for _ in range(k)]
    for bucket_idx, group in enumerate(scaffold_sets):
        buckets[bucket_idx % k].extend(group)

    test_idx = set(buckets[fold])
    val_idx  = set(buckets[(fold + 1) % k])
    train_idx = [i for i in range(len(smiles_list)) if i not in test_idx and i not in val_idx]
    return train_idx, list(val_idx), list(test_idx)
