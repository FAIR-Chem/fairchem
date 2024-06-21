import torch
import torch.nn as nn

from .mlp import ResMLP
from .sparse_att import SparseScaledDotProduct, Projection, _from_coo
    
class PositionFeaturizer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.,
        att_dropout: float = 0.,
        num_heads: int = 8,
    ):
        super().__init__()

        self.query_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        
        self.key_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.attention = SparseScaledDotProduct(
            dropout=att_dropout
        )

        self.mlp = ResMLP(
            input_dim=embed_dim+3*num_heads,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.norm_att = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)

        self.num_heads = num_heads

    def forward(
            self, 
            x: torch.Tensor,
            row_index: torch.Tensor,
            src_index: torch.Tensor,
            att_bias: torch.Tensor,
            pos: torch.Tensor,
            src_pos: torch.Tensor,
            org_to_src: torch.Tensor,
        ):

        # normalize inputs
        z_att = self.norm_att(x)

        # prepare inputs to attention
        query = self.query_proj(z_att)
        key = self.key_proj(z_att)[:, org_to_src]
        value = src_pos.expand(self.num_heads, -1, -1)

        # construct CSR format mask
        mask = _from_coo(pos.size(0), src_pos.size(0), row_index, src_index, att_bias)

        # compute scaled dot product attention
        feat, _ = self.attention(query, key, value, mask)

        # subtract positions to get displacements
        feat = feat - pos.expand(self.num_heads, -1, -1)

        # combine batched dimensions
        feat = feat.permute(1, 0, 2).reshape(x.size(0), -1)

        # normalize for mlp
        z_mlp = self.norm_mlp(x)

        # output with residual connection
        x = x + self.mlp(torch.cat([z_mlp, feat], dim=-1))

        return x