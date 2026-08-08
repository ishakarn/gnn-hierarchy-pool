"""GNN encoders (GCN, GIN, GAT, GraphSAGE) + readout + MLP classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv

from src.readouts import get_readout


def _mlp_head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Two-layer MLP classifier head."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class GCNModel(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        readout_type: str,
        num_tasks: int = 1,
        **_,
    ):
        super().__init__()
        dims = [num_node_features] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout
        self.readout = get_readout(readout_type, hidden_dim)
        self.classifier = _mlp_head(hidden_dim, hidden_dim // 2, num_tasks, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.readout(x, batch)          # [num_graphs, hidden_dim]
        return self.classifier(x)           # [num_graphs, num_tasks]


class GraphSAGEModel(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        readout_type: str,
        num_tasks: int = 1,
        **_,
    ):
        super().__init__()
        dims = [num_node_features] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout
        self.readout = get_readout(readout_type, hidden_dim)
        self.classifier = _mlp_head(hidden_dim, hidden_dim // 2, num_tasks, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.readout(x, batch)
        return self.classifier(x)


class GINModel(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        readout_type: str,
        num_tasks: int = 1,
        **_,
    ):
        super().__init__()
        self.convs = nn.ModuleList()

        # First layer: num_node_features -> hidden_dim
        self.convs.append(GINConv(
            nn.Sequential(
                nn.Linear(num_node_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            train_eps=True,
        ))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                train_eps=True,
            ))

        self.dropout = dropout
        self.readout = get_readout(readout_type, hidden_dim)
        self.classifier = _mlp_head(hidden_dim, hidden_dim // 2, num_tasks, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.readout(x, batch)
        return self.classifier(x)


class GATModel(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        readout_type: str,
        num_tasks: int = 1,
        num_heads: int = 4,
        **_,
    ):
        super().__init__()
        # Use hidden_dim // num_heads per head so concat output == hidden_dim.
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        head_dim = hidden_dim // num_heads

        self.convs = nn.ModuleList()
        # First layer input is num_node_features; all subsequent inputs are hidden_dim.
        in_dims = [num_node_features] + [hidden_dim] * (num_layers - 1)
        for in_d in in_dims:
            self.convs.append(
                GATConv(in_d, head_dim, heads=num_heads, dropout=dropout, concat=True)
            )

        self.dropout = dropout
        self.readout = get_readout(readout_type, hidden_dim)
        self.classifier = _mlp_head(hidden_dim, hidden_dim // 2, num_tasks, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))   # ELU is standard for GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.readout(x, batch)
        return self.classifier(x)

class GCNPModel(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        readout_type: str,
        num_tasks: int = 1,
        **_,
    ):
        super().__init__()
        dims = [num_node_features] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout
        self.readout = get_readout(readout_type, hidden_dim)
        self.classifier = _mlp_head(hidden_dim*len(self.convs), hidden_dim*len(self.convs) // 2, num_tasks, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        readout_pyramid = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            read = self.readout(x, batch)          # [num_graphs, hidden_dim]
            readout_pyramid.append(read)
        
        x = torch.cat(readout_pyramid, dim=1)
        return self.classifier(x)           # [num_graphs, num_tasks]

MODEL_REGISTRY = {
    'gcn':       GCNModel,
    'graphsage': GraphSAGEModel,
    'gin':       GINModel,
    'gat':       GATModel,
    'gcnp':      GCNPModel
}


def build_model(model_name: str, num_node_features: int, cfg: dict, num_tasks: int = 1) -> nn.Module:
    """Instantiate a model from the registry using a config dict."""
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](
        num_node_features=num_node_features,
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
        readout_type=cfg['readout'],
        num_tasks=num_tasks,
        num_heads=cfg.get('num_heads', 4),
    )
