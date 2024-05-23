import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import ResMLP

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

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.feed_forward = ResMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None, 
        attn_mask: torch.Tensor = None
    ) -> torch.tensor:
        """
        transform the input using the attention block
        arguments:
            x: input sequence of shape (L, N, C)
            padding_mask: a mask of shape (N, L) with `True` represents padding
        returns:
            transformed sequence of shape (L, N, C)
        """

        z = self.norm_attn(x)
        self_attn, *_ = self.self_attn(z, z, z, key_padding_mask=padding_mask, attn_mask=attn_mask)
        x = x + self.dropout(self_attn)
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        
        return x