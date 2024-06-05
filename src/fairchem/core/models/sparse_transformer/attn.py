import math
from typing import Optional

import torch
import torch.nn as nn

from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

from .mlp import ResMLP

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
        output = output.view(index.size(0), self.num_heads, self.d_k)

        return output

class SparseSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.,
        use_vec_hat: bool = False,
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

        if use_vec_hat:
            self.vec_hat_proj = nn.Linear(
                3, d_k * num_heads, False
            )

        self.output = nn.Linear(
            d_k * num_heads, embed_dim
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(d_k)
        self.use_vec_hat = use_vec_hat
        self.num_heads = num_heads
        self.d_k = d_k

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        vec_hat: Optional[torch.Tensor] = None,
        need_weights: Optional[bool] = True,
    ):
        # project to qkv
        query = self.query_proj(x, edge_index[0])
        key = self.key_proj(x, edge_index[1])
        value = self.value_proj(x, edge_index[1])

        if self.use_vec_hat and vec_hat is not None:
            value += self.vec_hat_proj(vec_hat).view(-1, self.num_heads, self.d_k)

        # compute scores
        logits = self.scale * (query * key).sum(2, keepdim=True)
        if attn_bias is not None:
            logits += attn_bias[..., None]

        scores = scatter_softmax(
            src = logits,
            index = edge_index[0],
            dim = 0
        )

        # dropout 
        scores = self.dropout(scores)

        # value aggregation
        self_attn = scatter(
            src = scores * value,
            index = edge_index[0],
            dim = 0,
            reduce = "sum",
            dim_size = x.size(0)
        )

        self_attn = self_attn.view(self_attn.size(0), -1)
        self_attn = self.output(self_attn)

        if need_weights:
            return self_attn, logits
        else:
            return self_attn

class SelfAttentionLayer(nn.Module):
    """
    An attention block implementation based on https://nn.labml.ai/transformers/models.html
    """
    def __init__(
        self, 
        embed_dim: int = 512, 
        hidden_dim: int = 1024,
        num_heads: int = 8, 
        dropout: float = 0,
        use_vec_hat: bool = False,
        activation: nn.Module = nn.SiLU(),
    ):
        """
        Initialize an `SelfAttentionLayer` instance
        arguments:
            embed_dim: the size of input 
            hidden_dim: hidden size of the feed forward network
            num_heads: number of heads used in MHA
            dropout: dropout strength
            activation: activation function
        """
        super().__init__()

        self.self_attn = SparseSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_vec_hat=use_vec_hat,
        )

        self.feed_forward = ResMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.norm_attn = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor, 
        attn_bias: Optional[torch.Tensor] = None,
        vec_hat: Optional[torch.Tensor] = None,
    ) -> torch.tensor:
        
        z = self.norm_attn(x)
        self_attn = self.self_attn(z, edge_index, attn_bias, vec_hat, False)
        x = x + self_attn
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + ff
        
        return x