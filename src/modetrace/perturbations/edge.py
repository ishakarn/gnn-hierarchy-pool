"""
Edge feature shuffle perturbations.

WEFS — Within-Graph Edge Feature Shuffle:
    For each graph independently, permute p-fraction of directed edge feature
    vectors among themselves (rows of g.edge_attr are randomly reassigned within
    the same graph).

GEFS — Global Edge Feature Shuffle:
    Pool all edge feature vectors across every graph in the split, permute
    p-fraction globally, then slice back.  Breaks edge-to-graph assignment.

Note: PyG molecular graphs store each bond as two directed edges (i→j and j→i)
with identical features.  These functions permute all directed edge feature rows
independently — bidirectionality is intentionally broken to maximally disrupt the
edge signal.
"""
from __future__ import annotations
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concat_edge_features(graphs: List[Data]):
    feats, sizes = [], []
    F = None
    for g in graphs:
        ea = getattr(g, "edge_attr", None)
        if ea is None or ea.numel() == 0:
            sizes.append(0)
            if F is None:
                F = 0
            continue
        feats.append(ea)
        sizes.append(ea.size(0))
        F = ea.size(1)
    X = torch.cat(feats, dim=0) if feats else torch.empty((0, F or 0))
    return X, sizes


def _split_back(X: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    F = X.size(1) if X.dim() == 2 else 0
    chunks, s = [], 0
    for n in sizes:
        if n == 0:
            chunks.append(torch.empty((0, F), dtype=X.dtype, device=X.device))
        else:
            chunks.append(X[s : s + n])
            s += n
    return chunks


# ---------------------------------------------------------------------------
# WEFS
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_WEFS(graphs: List[Data], p: float, seed: int) -> List[Data]:
    """Within-graph edge feature shuffle at rate p."""
    assert 0.0 <= p <= 1.0
    out = []
    for i, g in enumerate(graphs):
        rng = np.random.default_rng(seed + 1_000_003 * i)
        ng = g.clone()
        ea = getattr(g, "edge_attr", None)
        if ea is None or ea.numel() == 0:
            out.append(ng)
            continue
        n = ea.size(0)
        k = int(np.floor(p * n))
        if k <= 1:
            out.append(ng)
            continue
        S    = torch.as_tensor(rng.choice(n, size=k, replace=False), device=ea.device, dtype=torch.long)
        perm = torch.as_tensor(rng.permutation(k),                   device=ea.device, dtype=torch.long)
        ea_new = ea.clone()
        ea_new[S] = ea[S][perm]
        ng.edge_attr = ea_new.contiguous()
        out.append(ng)
    return out


# ---------------------------------------------------------------------------
# GEFS
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_GEFS(graphs: List[Data], p: float, seed: int) -> List[Data]:
    """Global edge feature shuffle at rate p across the whole split."""
    assert 0.0 <= p <= 1.0
    rng = np.random.default_rng(seed)
    new_graphs = [g.clone() for g in graphs]

    X_all, sizes = _concat_edge_features(graphs)
    M = X_all.size(0)
    if M <= 1:
        return new_graphs

    k = int(np.floor(p * M))
    if k <= 1:
        return new_graphs

    S    = torch.as_tensor(rng.choice(M, size=k, replace=False), device=X_all.device, dtype=torch.long)
    perm = torch.as_tensor(rng.permutation(k),                   device=X_all.device, dtype=torch.long)
    Xp   = X_all.clone()
    Xp[S] = X_all[S][perm]

    chunks = _split_back(Xp, sizes)
    for i, g in enumerate(graphs):
        ea = getattr(g, "edge_attr", None)
        if ea is not None and ea.numel() > 0:
            new_graphs[i].edge_attr = chunks[i].to(dtype=ea.dtype)
    return new_graphs
