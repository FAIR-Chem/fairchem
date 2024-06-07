import math
from typing import Optional

import torch
import torch.nn as nn

from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

class Projection(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512, 
        num_heads: int = 8,
    ):
        super().__init__()

        self.d_k = embed_dim // num_heads
        self.linear = nn.Linear(embed_dim, num_heads * self.d_k)
        self.num_heads = num_heads
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.)

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
    ):
        output = self.linear(x)[index]
        output = output.reshape(index.size(0), self.num_heads, self.d_k)

        return output

class ScatterSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.,
    ):
        super().__init__()

        d_k = embed_dim // num_heads

        self.query_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
        self.key_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.value_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.output = nn.Linear(
            d_k * num_heads, embed_dim
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(d_k)
        self.num_heads = num_heads
        self.d_k = d_k

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        att_bias: Optional[torch.Tensor] = None,
        need_weights: Optional[bool] = True,
    ):
        # project to qkv
        query = self.query_proj(x, edge_index[0])
        key = self.key_proj(x, edge_index[1])
        value = self.value_proj(x, edge_index[1])

        # compute scores
        logits = self.scale * (query * key).sum(2, keepdim=True)
        if att_bias is not None:
            logits += att_bias[..., None]

        scores = scatter_softmax(
            src = logits,
            index = edge_index[0],
            dim = 0
        )

        # dropout 
        scores = self.dropout(scores)

        # value aggregation
        self_att = scatter(
            src = scores * value,
            index = edge_index[0],
            dim = 0,
            reduce = "sum",
            dim_size = x.size(0)
        )

        self_att = self_att.reshape(self_att.size(0), -1)
        self_att = self.output(self_att)

        if need_weights:
            return self_att, logits
        else:
            return self_att