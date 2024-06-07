import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .mlp import ResMLP
from .sparse_att import SparseSelfAttention
from .scatter_att import ScatterSelfAttention

class AttentionLayer(nn.Module):
    """
    An attention block implementation based on https://nn.labml.ai/transformers/models.html
    """
    def __init__(
        self, 
        embed_dim: int = 512, 
        hidden_dim: int = 1024,
        num_heads: int = 8, 
        dropout: float = 0.,
        att_dropout: float = 0.,
        attention: str = "sparse",
        activation: nn.Module = nn.SiLU(),
    ):
        """
        Initialize an `SelfAttentionLayer` instance
        arguments:
            embed_dim: the size of input 
            hidden_dim: hidden size of the feed forward network
            num_heads: number of heads used in MHA
            dropout: dropout strength
            attention: can be "flash", "sparse", or "scatter"
            activation: activation function
        """
        super().__init__()

        if attention == "flash":
            self.self_att = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=att_dropout
            )
        elif attention == "sparse":
            self.self_att = SparseSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=att_dropout
            )
        elif attention == "scatter":
            self.self_att = ScatterSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=att_dropout
            )
        else:
            raise NotImplementedError("The attention mechanism specified is not implemented yet!")
        
        self.feed_forward = ResMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.norm_att = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.activation = activation
        self.attention = attention

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        att_bias: Optional[torch.Tensor] = None,
        att_mask: Optional[torch.Tensor] = None,
    ) -> torch.tensor:
        """
        transform the input using the attention block
        arguments:
            x: input sequence of shape (L, B, C) or (L, C)
            edge_index: coo formated sparse matrix of shape (2, E)
            att_bias: a tensor of shape (E, H)
            att_mask: a mask of shape (B * H, L, L)
        """

        z = self.norm_att(x)
        
        if self.attention == "flash":
            self_att, _ = self.self_att(
                z, z, z, att_mask=att_mask, need_weights=False
            )
        else:
            assert edge_index is not None
            self_att = self.self_att(
                z, edge_index=edge_index, att_bias=att_bias, need_weights=False
            )

        x = x + self_att
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + ff
        
        return x