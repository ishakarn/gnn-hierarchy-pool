"""GNN encoders (GCN, GIN, GINE, GAT, GraphSAGE, GCN-P, AttentiveFP) + readout + MLP classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATConv, SAGEConv
from torch_geometric.nn.models.attentive_fp import GATEConv

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
        readout_kwargs: dict = None,
        **_,
    ):
        super().__init__()
        dims = [num_node_features] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout
        self.readout = get_readout(readout_type, hidden_dim, **(readout_kwargs or {}))
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
        readout_kwargs: dict = None,
        **_,
    ):
        super().__init__()
        dims = [num_node_features] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout
        self.readout = get_readout(readout_type, hidden_dim, **(readout_kwargs or {}))
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
        readout_kwargs: dict = None,
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
        self.readout = get_readout(readout_type, hidden_dim, **(readout_kwargs or {}))
        self.classifier = _mlp_head(hidden_dim, hidden_dim // 2, num_tasks, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.readout(x, batch)
        return self.classifier(x)


class GINEModel(nn.Module):
    """GIN with edge features (GINEConv); the only encoder here that reads edge_attr.
    Without edge features it uses zero edge embeddings."""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        readout_type: str,
        num_tasks: int = 1,
        num_edge_features: int = 0,
        **_,
    ):
        super().__init__()
        self.num_edge_features = num_edge_features
        self.node_proj = nn.Linear(num_node_features, hidden_dim)
        self.edge_proj = nn.Linear(max(num_edge_features, 1), hidden_dim)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            for _ in range(num_layers)
        ])
        self.dropout = dropout
        self.readout = get_readout(readout_type, hidden_dim)
        self.classifier = _mlp_head(hidden_dim, hidden_dim // 2, num_tasks, dropout)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = F.relu(self.node_proj(x))

        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is not None and self.num_edge_features > 0:
            e = F.relu(self.edge_proj(edge_attr.float()))
        else:
            e = F.relu(self.edge_proj(x.new_zeros(edge_index.size(1), 1)))

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, e))
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
        readout_kwargs: dict = None,
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
        self.readout = get_readout(readout_type, hidden_dim, **(readout_kwargs or {}))
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
        readout_kwargs: dict = None,
        **_,
    ):
        super().__init__()
        dims = [num_node_features] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout = dropout
        self.readout = get_readout(readout_type, hidden_dim, **(readout_kwargs or {}))
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

class AttentiveFPModel(nn.Module):
    """AttentiveFP atom encoder (Xiong et al., J. Med. Chem. 2019) + any readout.

    The atom-embedding layers are built from PyG's AttentiveFP components and
    wired exactly as in torch_geometric.nn.models.AttentiveFP.forward. The
    molecule-level part is the swappable readout: readout='afp' reproduces
    PyG's AttentiveFP; mean/sum/max/attention/selfattention replace it.

    head='mlp' uses the repo's 2-layer MLP classifier (same as other models);
    head='linear' uses PyG's original predictor (dropout -> Linear).
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        readout_type: str,
        num_tasks: int = 1,
        num_edge_features: int = 0,
        head: str = 'mlp',
        readout_kwargs: dict = None,
        **_,
    ):
        super().__init__()
        if not num_edge_features:
            raise ValueError("AttentiveFP needs bond features (num_edge_features > 0)")
        self.dropout = dropout

        self.lin1 = nn.Linear(num_node_features, hidden_dim)
        self.gate_conv = GATEConv(hidden_dim, hidden_dim, num_edge_features, dropout)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.atom_convs = nn.ModuleList()
        self.atom_grus = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.atom_convs.append(GATConv(hidden_dim, hidden_dim, dropout=dropout,
                                           add_self_loops=False, negative_slope=0.01))
            self.atom_grus.append(nn.GRUCell(hidden_dim, hidden_dim))

        self.readout = get_readout(readout_type, hidden_dim, **(readout_kwargs or {}))

        self.head = head
        if head == 'mlp':
            self.classifier = _mlp_head(hidden_dim, hidden_dim // 2, num_tasks, dropout)
        elif head == 'linear':
            self.classifier = nn.Linear(hidden_dim, num_tasks)
        else:
            raise ValueError(f"Unknown head '{head}'. Choose from: mlp, linear")

    def encode(self, data):
        """Atom embeddings, [total_nodes, hidden_dim]."""
        x, edge_index = data.x.float(), data.edge_index
        edge_attr = data.edge_attr.float()

        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()
        return x

    def forward(self, data):
        x = self.readout(self.encode(data), data.batch)   # [num_graphs, hidden_dim]
        if self.head == 'linear':
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)                          # [num_graphs, num_tasks]


MODEL_REGISTRY = {
    'gcn':       GCNModel,
    'graphsage': GraphSAGEModel,
    'gin':       GINModel,
    'gine':      GINEModel,
    'gat':       GATModel,
    'gcnp':      GCNPModel,
    'attentivefp': AttentiveFPModel,
}


def build_model(model_name: str, num_node_features: int, cfg: dict, num_tasks: int = 1,
                num_edge_features: int = 0) -> nn.Module:
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
        num_edge_features=num_edge_features,
        head=cfg.get('head', 'mlp'),
        readout_kwargs={
            'num_timesteps': cfg.get('num_timesteps', 2),
            'dropout': cfg['dropout'],
        },
    )
