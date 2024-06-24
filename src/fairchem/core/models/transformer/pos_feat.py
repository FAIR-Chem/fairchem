import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import masked_matmul

from .mlp import ResMLP
from .sparse_att import Projection, _from_coo, _wrap_value
    
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
            input_dim=embed_dim+4*num_heads,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            dropout=dropout,
        )

        self.norm_att = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.norm_feat = nn.BatchNorm1d(4*num_heads)

        self.num_heads = num_heads

    def forward(
            self, 
            x: torch.Tensor,
            edge_index: torch.Tensor,
            src_index: torch.Tensor,
            edge_to_src: torch.Tensor,
            att_bias: torch.Tensor,
            dist: torch.Tensor,
            pos: torch.Tensor,
            src_pos: torch.Tensor,
        ):

        # normalize inputs
        z_att = self.norm_att(x)

        # prepare inputs to attention
        query = self.query_proj(z_att)
        key = self.key_proj(z_att)
        value = torch.cat([
            src_pos.expand(self.num_heads, -1, -1),
            torch.ones((self.num_heads, src_pos.size(0), 1), device=src_pos.device)
        ], dim=-1)

        # construct CSR format mask
        dummy = torch.zeros((self.num_heads, edge_index.size(1)), device=edge_index.device)
        mask = _from_coo(query.size(1), key.size(1), edge_index[0], edge_index[1], dummy)

        # scale before attend
        query = query / math.sqrt(query.size(-1))
        
        # this only takes care of QK^T, bias must be added manually.
        logits = masked_matmul(query, key.transpose(-2, -1), mask)

        # move to src graph
        if edge_to_src is not None:
            logits = logits.values()[:, edge_to_src] + att_bias
        else:
            logits = logits.values() + att_bias

        logits = _from_coo(query.size(1), value.size(1), src_index[0], src_index[1], logits)

        # softmax to get the attention probabilities
        # att = F.softmax(logits, dim=-1)
        att = _wrap_value(logits, F.sigmoid(logits.values()))

        # optional dropout
        att = self.att_drop(att)

        # normalize att
        att = _wrap_value(att, att.values() / dist.masked_fill(dist==0, torch.inf).view(1, -1))

        # aggregate
        feat = torch.bmm(att, value)

        # subtract positions to get displacements
        feat[:, :, :3] -= feat[:, :, 3:] * pos.expand(self.num_heads, -1, -1)

        # combine batched dimensions
        feat = feat.permute(1, 0, 2).reshape(x.size(0), -1)

        # normalize for mlp
        z_mlp = self.norm_mlp(x)
        feat = self.norm_feat(feat)

        # output with residual connection
        x = x + self.mlp(torch.cat([z_mlp, feat], dim=-1))

        return x