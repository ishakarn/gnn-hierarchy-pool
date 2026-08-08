"""Graph-level readout modules."""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
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


def get_readout(readout_type: str, hidden_dim: int = None):
    """Return a readout callable or nn.Module.

    For mean/sum/max: returns a function (x, batch) -> graph_emb.
    For attention:    returns a GatedAttentionReadout module.

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
    elif readout_type == 'selfattention':
        if hidden_dim is None:
            raise ValueError("hidden_dim is required for attention readout")
        return GraphSelfAttention(input_dim=hidden_dim, inner_dim=5*hidden_dim, mode='sample')
    else:
        raise ValueError(
            f"Unknown readout '{readout_type}'. "
            "Choose from: mean, sum, max, attention"
        )
        
