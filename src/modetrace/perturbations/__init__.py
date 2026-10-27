"""
Perturbation dispatcher.

Public interface:
    apply_perturbation(graphs, perturbation, p, seed) -> List[Data]

Canonical perturbation names:
    none  — no perturbation (returns clones)
    WNFS  — within-graph node feature shuffle
    GNFS  — global node feature shuffle
    WEFS  — within-graph edge feature shuffle  (molecular only)
    GEFS  — global edge feature shuffle        (molecular only)
    JGFS  — joint global node+edge shuffle     (molecular only, no within variant)
"""
from __future__ import annotations
from typing import List
from torch_geometric.data import Data

from .node  import apply_WNFS, apply_GNFS
from .edge  import apply_WEFS, apply_GEFS
from .joint import apply_JGFS


def apply_perturbation(
    graphs: List[Data],
    perturbation: str,
    p: float,
    seed: int,
) -> List[Data]:
    """
    Apply a named perturbation to a list of graphs.

    Args:
        graphs:      List of PyG Data objects (already cloned from the dataset).
        perturbation: Canonical name — one of {none, WNFS, GNFS, WEFS, GEFS, JGFS}.
        p:           Perturbation rate in [0, 1].  p=0 always returns clean clones.
        seed:        Integer seed for reproducibility.

    Returns:
        New list of Data objects with perturbation applied in-place on clones.
    """
    if perturbation == "none" or p == 0.0:
        return [g.clone() for g in graphs]

    dispatch = {
        "WNFS": apply_WNFS,
        "GNFS": apply_GNFS,
        "WEFS": apply_WEFS,
        "GEFS": apply_GEFS,
        "JGFS": apply_JGFS,
    }

    if perturbation not in dispatch:
        raise ValueError(
            f"Unknown perturbation '{perturbation}'. "
            f"Valid options: none, WNFS, GNFS, WEFS, GEFS, JGFS"
        )

    return dispatch[perturbation](graphs, p=p, seed=seed)
