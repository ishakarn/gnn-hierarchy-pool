"""Graph-level readout modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.aggr import SetTransformerAggregation
from torch_geometric.utils import softmax as pyg_softmax
from .pooling.attn_pooling import GraphSelfAttention, GraphMultiHeadSelfAttention


class GatedAttentionReadout(nn.Module):
    """Learnable graph-level gated attention readout.

    A two-layer gate network maps node embeddings -> scalar logits.
    Softmax is applied within each graph (using PyG's scatter softmax),
    then we return a weighted sum of node embeddings per graph.

    Works with batched PyG graphs via the `batch` index vector.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     [total_nodes, hidden_dim]  node embeddings
            batch: [total_nodes]              graph assignment (0-indexed)
        Returns:
            out:   [num_graphs, hidden_dim]   graph-level representations
        """
        # [total_nodes, 1] -> [total_nodes]
        logits = self.gate(x).squeeze(-1)

        # Softmax within each graph; pyg_softmax handles the scatter correctly
        attn_weights = pyg_softmax(logits, batch)  # [total_nodes]

        # Weighted sum: broadcast weights over feature dim
        # attn_weights: [total_nodes] -> [total_nodes, 1]
        num_graphs = int(batch.max().item()) + 1
        out = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
        out.scatter_add_(
            0,
            batch.unsqueeze(-1).expand_as(x),
            attn_weights.unsqueeze(-1) * x,
        )
        return out  # [num_graphs, hidden_dim]


READOUTS = ('mean', 'sum', 'max', 'attention', 'multihead', 'selfattention', 'afp')


class AttentiveFPReadout(nn.Module):
    """AttentiveFP molecule-level readout (Xiong et al., J. Med. Chem. 2019).

    A virtual super-node per graph starts as the ReLU'd sum of its node
    embeddings, then for `num_timesteps` rounds attends over the graph's
    nodes (GATConv from nodes -> super-node) and is updated by a GRU.

    Mirrors the "Molecule Embedding" block of
    torch_geometric.nn.models.AttentiveFP.forward, as a standalone readout
    usable after any encoder.
    """

    def __init__(self, hidden_dim: int, num_timesteps: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.mol_conv = GATConv(hidden_dim, hidden_dim, dropout=dropout,
                                add_self_loops=False, negative_slope=0.01)
        self.mol_conv.explain = False  # same as PyG: cannot explain global pooling
        self.mol_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Edges from every node to its graph's super-node
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for _ in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()
        return out  # [num_graphs, hidden_dim]


def get_readout(readout_type: str, hidden_dim: int = None,
                num_timesteps: int = 2, dropout: float = 0.0):
    """Return a readout callable or nn.Module.

    For mean/sum/max: returns a function (x, batch) -> graph_emb.
    For attention:    returns a GatedAttentionReadout module.
    For afp:          returns an AttentiveFPReadout module (uses
                      num_timesteps and dropout).

    The returned object is always called as readout(x, batch).
    """
    if readout_type == 'mean':
        return global_mean_pool
    elif readout_type == 'sum':
        return global_add_pool
    elif readout_type == 'max':
        return global_max_pool
    elif readout_type == 'attention':
        if hidden_dim is None:
            raise ValueError("hidden_dim is required for attention readout")
        return GatedAttentionReadout(hidden_dim)
    elif readout_type == 'multihead':
        if hidden_dim is None:
            raise ValueError("hidden_dim is required for multihead readout")
        return SetTransformerAggregation(hidden_dim, num_seed_points=1, heads=4)
    elif readout_type == 'selfattention':
        if hidden_dim is None:
            raise ValueError("hidden_dim is required for attention readout")
        return GraphSelfAttention(input_dim=hidden_dim, inner_dim=5*hidden_dim, mode='sample')
    elif readout_type == 'afp':
        if hidden_dim is None:
            raise ValueError("hidden_dim is required for afp readout")
        return AttentiveFPReadout(hidden_dim, num_timesteps=num_timesteps, dropout=dropout)
    else:
        raise ValueError(
            f"Unknown readout '{readout_type}'. "
            f"Choose from: {', '.join(READOUTS)}"
        )
        
