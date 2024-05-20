from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

from .utils import make_mlp

class TokenGNNEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        dropout: float = 0.,
        num_heads: int = 8,
    ):
        super().__init__()

        assert (embed_dim % num_heads) == 0

        self.query = Projection(embed_dim=embed_dim, num_heads=num_heads)
        self.key = Projection(embed_dim=embed_dim, num_heads=num_heads)
        self.value = Projection(embed_dim=embed_dim, num_heads=num_heads)

        self.output = nn.Linear(embed_dim, embed_dim)
        self.scale = 1 / math.sqrt(embed_dim // num_heads)

        self.feed_forward = make_mlp(
            input_dim=embed_dim,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        edges: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ):
        
        # projection
        z = self.attn_norm(x)
        query = self.key(z, edges[0], masks)
        key = self.key(z, edges[1], masks)
        value = self.key(z, edges[1], masks) # (E, H, d_k)

        # softmax
        logits = (query * key).sum(2, keepdim=True) # (E, H, 1)
        logits = logits - scatter(logits.detach(), edges[0], dim=0, reduce="max")[edges[0]] # (E, H, 1)
        scores = torch.exp(logits) # (E, H, 1)
        scores = scores / scatter(scores, edges[0], dim=0, reduce="sum")[edges[0]] # (E, H, 1)

        # dropout 
        scores = self.dropout(scores)

        # value aggregation
        self_attn = scatter(scores * value, edges[0], dim=0, reduce="sum") # (L, H, d_k)
        self_attn = self_attn.view(self_attn.size(0), -1)

        # residual connection
        x = x + self.dropout(self_attn)

        # feed forward and residual connection
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)

        return x
    
class Projection(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512, 
        num_heads: int = 8,
    ):
        super().__init__()
        d_k = embed_dim // num_heads
        self.linears = nn.ModuleList([
            nn.Linear(embed_dim, num_heads * d_k)
            for _ in range(5)
        ])
        self.heads = num_heads
        self.d_k = d_k

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
        masks: torch.Tensor
    ):
        output = torch.zeros((index.size(0), self.heads * self.d_k), device=x.device)
        for mask, linear in zip(masks, self.linears):
            output[mask] = linear(x[index[mask]])
        output = output.view(index.size(0), self.heads, self.d_k)

        return output


class OutputModule(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        dropout: float = 0,
    ):
        super().__init__()

        self.energy_output = make_mlp(
            3 * embed_dim, ff_dim, 1, dropout
        )

        self.forces_output = make_mlp(
            3 * embed_dim, ff_dim, 1, dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        # prepare inputs
        edge_token_pos = torch.arange(pos.size(0), pos.size(0) + edge_index.size(1), device=edge_index.device)
        inputs = torch.cat([x[edge_index[0]], x[edge_index[1]], x[edge_token_pos]], dim = -1)
        vec = pos[edge_index[0]] - pos[edge_index[1]]
        vec_hat = F.normalize(vec, dim = -1)

        # regress energy and forces
        energy_pairs = self.energy_output(inputs)
        force_pairs = self.forces_output(inputs) * vec_hat
        energy = scatter(energy_pairs, batch[edge_index[0]], dim = 0, reduce="sum")
        forces = scatter(force_pairs, edge_index[0], dim = 0, reduce="sum")

        return energy, forces
