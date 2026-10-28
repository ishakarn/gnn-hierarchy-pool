"""
Node feature shuffle perturbations.

WNFS — Within-Graph Node Feature Shuffle:
    For each graph independently, permute p-fraction of node feature vectors
    among themselves (rows of g.x are randomly reassigned within the same graph).

GNFS — Global Node Feature Shuffle:
    Pool all node feature vectors across every graph in the split into a single
    matrix, permute p-fraction of those rows globally, then slice back into
    the original graphs.  This breaks the node-to-graph assignment.
"""
from __future__ import annotations
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concat_node_features(graphs: List[Data]):
    feats, sizes = [], []
    F = None
    for g in graphs:
        x = getattr(g, "x", None)
        if x is None or x.numel() == 0:
            sizes.append(0)
            if F is None:
                F = 0
            continue
        feats.append(x)
        sizes.append(x.size(0))
        F = x.size(1)
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
# WNFS
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_WNFS(graphs: List[Data], p: float, seed: int) -> List[Data]:
    """Within-graph node feature shuffle at rate p."""
    assert 0.0 <= p <= 1.0
    out = []
    for i, g in enumerate(graphs):
        rng = np.random.default_rng(seed + 1_000_003 * i)
        ng = g.clone()
        x = getattr(g, "x", None)
        if x is None or x.numel() == 0:
            out.append(ng)
            continue
        n = x.size(0)
        k = int(np.floor(p * n))
        if k <= 1:
            out.append(ng)
            continue
        S    = torch.as_tensor(rng.choice(n, size=k, replace=False), device=x.device, dtype=torch.long)
        perm = torch.as_tensor(rng.permutation(k),                   device=x.device, dtype=torch.long)
        x_new = x.clone()
        x_new[S] = x[S][perm]
        ng.x = x_new.contiguous()
        out.append(ng)
    return out


# ---------------------------------------------------------------------------
# GNFS
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_GNFS(graphs: List[Data], p: float, seed: int) -> List[Data]:
    """Global node feature shuffle at rate p across the whole split."""
    assert 0.0 <= p <= 1.0
    rng = np.random.default_rng(seed)
    new_graphs = [g.clone() for g in graphs]

    X_all, sizes = _concat_node_features(graphs)
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

    # _split_back returns one chunk per graph (empty graphs included), so index
    # by i. (The original indexed by a counter that skipped empty graphs, which
    # misassigned features after the first graph with no nodes.)
    chunks = _split_back(Xp, sizes)
    for i, g in enumerate(graphs):
        x = getattr(g, "x", None)
        if x is not None and x.numel() > 0:
            new_graphs[i].x = chunks[i].to(dtype=x.dtype)
    return new_graphs
