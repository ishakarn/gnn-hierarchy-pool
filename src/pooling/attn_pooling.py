import torch
from torch.nn import Linear

from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
import math
# torch.autograd.set_detect_anomaly(True, check_nan=False)

def graph_attn_op_batched(q, k, v, batch, batch_size):
    q, mask = to_dense_batch(q, batch)
    k, _    = to_dense_batch(k, batch)
    v, _    = to_dense_batch(v, batch)


    attn_maps = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1]) # (batch, padded_num_nodes, padded_num_nodes)
    attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    attn_maps = attn_maps.masked_fill(~(attn_mask), -torch.inf)
    attn_maps = F.softmax(attn_maps, dim=-1)
    attn_maps = attn_maps.masked_fill(torch.isnan(attn_maps), 0)

    # print(attn_maps[0])

    # print(attn_maps[0])

    return torch.bmm(attn_maps, v)

class GraphSelfAttention(torch.nn.Module):
    def __init__(self, input_dim, inner_dim, mode='sample'):
        super().__init__()

        assert mode in ['sample', 'mean']
        self.mode = mode
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.q = torch.nn.Linear(input_dim, inner_dim)
        self.k = torch.nn.Linear(input_dim, inner_dim)
        self.v = torch.nn.Linear(input_dim, inner_dim)
        self.out_proj = torch.nn.Linear(inner_dim, input_dim, bias=False)
    
    def forward(self, x, batch):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        out = graph_attn_op_batched(q, k, v, batch, batch.max() + 1)
        out = self.out_proj(out)
        
        if self.mode == 'sample':
            return out[:, 0, :]
        elif self.mode == 'mean':
            is_not_zero = out.abs() > 1e-5 # hardcoded tolerance
            is_not_padding = is_not_zero.sum(dim=-1) > 0
            num_not_padding = is_not_padding.sum(dim=-1)

            return out.sum(dim=-2) / num_not_padding.unsqueeze(-1)

class GraphMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, input_dim, inner_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.heads = torch.nn.ModuleList([
            GraphSelfAttention(input_dim, inner_dim) for _ in range(num_heads)
        ])
        self.out_proj = torch.nn.Linear(inner_dim, input_dim, bias=False)

    def forward(self, x, batch):
        head_outputs = [head.forward(x, batch) for head in self.heads]
        # Concatenate along the last dimension
        out = torch.cat(head_outputs, dim=-1)
        out = self.out_proj(out)

        if self.mode == 'sample':
            return out[:, 0, :]
        elif self.mode == 'mean':
            '''
            Tries to figure out the true shape by guessing the padding
            '''
            is_not_zero = out.abs() > 1e-5 # hardcoded tolerance
            is_not_padding = is_not_zero.sum(dim=-1) > 0
            num_not_padding = is_not_padding.sum(dim=-1)

            return out.sum(dim=-2) / num_not_padding.unsqueeze(-1)