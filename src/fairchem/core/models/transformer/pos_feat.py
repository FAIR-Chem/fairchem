import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import masked_matmul

from .mlp import ResMLP
from .sparse_att import SparseScaledDotProduct, Projection, _from_coo, _wrap_value
    
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

        self.att_drop = nn.Dropout(att_dropout)

        self.mlp = ResMLP(
            input_dim=embed_dim+3*num_heads,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.norm_att = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(
            self.mlp.input.weight[:, :self.embed_dim],
            -math.sqrt(3 / self.embed_dim),
            math.sqrt(3 / self.embed_dim)
        )
        nn.init.uniform_(
            self.mlp.input.weight[:, self.embed_dim:],
            -math.sqrt(1 / self.embed_dim),
            math.sqrt(1 / self.embed_dim)
        )

    def forward(
            self, 
            x: torch.Tensor,
            row_index: torch.Tensor,
            src_index: torch.Tensor,
            att_bias: torch.Tensor,
            dist: torch.Tensor,
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

        # construct CSR format mask and normalizer
        mask = _from_coo(pos.size(0), src_pos.size(0), row_index, src_index, att_bias)
        norm = _wrap_value(mask, 1 / dist.masked_fill(dist==0, torch.inf).expand(self.num_heads, -1).clone())

        # Attend: (B x nh, S, hs) x (B x nh, hs, S) -> (B x nh, S, S)
        query = query / math.sqrt(query.size(-1))
        
        # this only takes care of QK^T, bias must be added manually.
        logits = masked_matmul(query, key.transpose(-2, -1), mask)
        logits = logits + mask

        # Softmax to get the attention probabilities
        att = F.softmax(logits, dim=-1)

        # normalize to get unit vectors
        att = att * norm

        # Optional dropout, could be part of the masking in the future
        att = _wrap_value(att, self.att_drop(att.values().clone()))

        # Get to the predicted values, for all heads
        feat = torch.bmm(att, value)
        norm = torch.bmm(att, torch.ones((self.num_heads, src_pos.size(0), 1), device=src_pos.device))

        # subtract positions to get displacements
        feat = feat - norm * pos.expand(self.num_heads, -1, -1)

        # combine batched dimensions
        feat = feat.permute(1, 0, 2).reshape(x.size(0), -1)

        # normalize for mlp
        z_mlp = self.norm_mlp(x)

        # output with residual connection
        x = x + self.mlp(torch.cat([z_mlp, feat], dim=-1))

        return x