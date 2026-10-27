"""
Joint Global Feature Shuffle (JGFS).

Shuffles both node and edge features globally at rate p.
There is no within-graph variant — only a global joint shuffle makes sense
for isolating the structural (topology-only) information mode.

Implemented by composing GNFS and GEFS with different seeds to ensure
the node and edge shuffles are independent random permutations.
"""
from __future__ import annotations
from typing import List
from torch_geometric.data import Data
from .node import apply_GNFS
from .edge import apply_GEFS


def apply_JGFS(graphs: List[Data], p: float, seed: int) -> List[Data]:
    """
    Joint global feature shuffle at rate p.
    Applies GNFS then GEFS with independent seeds (seed and seed+1).
    """
    graphs = apply_GNFS(graphs, p=p, seed=seed)
    graphs = apply_GEFS(graphs, p=p, seed=seed + 1)
    return graphs
