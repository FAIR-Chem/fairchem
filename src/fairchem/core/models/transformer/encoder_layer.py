import math
import torch
import torch.nn as nn

from .mlp import ResMLP
from .sparse_att import SparseSelfAttention
from .pos_feat import PositionFeaturizer

class EncoderLayer(nn.Module):
    """
    An encoder layer implementation based on https://nn.labml.ai/transformers/models.html
    """
    def __init__(
        self, 
        embed_dim: int = 512, 
        hidden_dim: int = 1024,
        num_heads: int = 8, 
        dropout: float = 0.,
        att_dropout: float = 0.,
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

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_att = SparseSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=att_dropout
        )

        self.pos_feat = PositionFeaturizer(
            embed_dim=embed_dim,
            att_dropout=att_dropout,
            num_heads=num_heads
        )

        self.feed_forward_att = ResMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.feed_forward_pos = ResMLP(
            input_dim=embed_dim+4*num_heads,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.norm_att = nn.LayerNorm(embed_dim)
        self.norm_pos = nn.LayerNorm(embed_dim)
        self.norm_ff_att = nn.LayerNorm(embed_dim)
        self.norm_ff_pos = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        nn.init.uniform_(
            self.feed_forward_pos.input.weight[:, :self.embed_dim],
            - math.sqrt(3 / self.embed_dim),
            math.sqrt(3 / self.embed_dim)
        )
        nn.init.uniform_(
            self.feed_forward_pos.input.weight[:, self.embed_dim:],
            - math.sqrt(3 / self.num_heads),
            math.sqrt(3 / self.num_heads)
        )

    def forward(
        self,
        x: torch.Tensor,
        row_index: torch.Tensor, 
        col_index: torch.Tensor,
        to_col_index: torch.Tensor,
        att_bias: torch.Tensor,
        pos_att_bias: torch.Tensor,
        dist: torch.Tensor,
        pos: torch.Tensor,
        col_pos: torch.Tensor,
    ) -> torch.tensor:
        """
        transform the input using the attention block
        arguments:
            x: input sequence of shape (L, C)
            row_index: coo formated sparse matrix of shape (E,), bounded by L
            col_index: coo formated sparse matrix of shape (E,), bounded by S
            to_col_index: coo formated sparse matrix of shape (E,), bounded by S'
            att_bias: a tensor of shape (E, H)
            pos_att_bias: a tensor of shape (E, H)
            dist: a tensor of shape (E,)
            pos: a tensor of shape (L, 3)
            col_pos: a tensor of shape (S', 3)
        """

        z = self.norm_att(x)
        
        self_att = self.self_att(
            z, row_index, col_index, to_col_index, att_bias, False
        )

        x = x + self_att

        z = self.norm_ff_att(x)
        ff = self.feed_forward_att(z)
        x = x + ff

        z = self.norm_pos(x)
        pos_feat = self.pos_feat(
            z,
            row_index,
            col_index,
            to_col_index,
            pos_att_bias,
            dist,
            pos,
            col_pos,
        )

        # concatenate here since residual is not well defined for new inputs
        z = self.norm_ff_pos(x)
        ff = self.feed_forward_pos(torch.cat([z, pos_feat], dim=-1))
        x = x + ff
        
        return x