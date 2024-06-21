import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import masked_matmul

from .sparse_att import Projection, _from_coo, _wrap_value
    
class EuclideanAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        att_dropout: float = 0.,
        num_heads: int = 8,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.query_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.query_vec_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_k=3,
        )
        
        self.key_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.value_proj = Projection(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.att_drop = nn.Dropout(att_dropout)

        self.num_heads = num_heads

        self.output = nn.Linear(
            embed_dim + num_heads * 4, embed_dim
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
            self, 
            x: torch.Tensor,
            row_index: torch.Tensor,
            col_index: torch.Tensor,
            to_col_index: torch.Tensor,
            att_bias: torch.Tensor,
            dist: torch.Tensor,
            pos: torch.Tensor,
            col_pos: torch.Tensor,
        ):

        # prepare inputs to attention
        query = self.query_proj(x)
        query_vec = self.query_vec_proj(x)
        if to_col_index is not None:
            key = self.key_proj(x)[:, to_col_index]
            value = self.value_proj(x)[:, to_col_index]
        else:
            key = self.key_proj(x)
            value = self.value_proj(x)
        dst_vec = col_pos.expand(self.num_heads, -1, -1)
        dist = dist.masked_fill(dist==0, torch.inf).view(1, -1)

        # construct CSR format mask
        mask = _from_coo(query.size(1), key.size(1), row_index, col_index, att_bias)
        
        # scale before attend
        query = query / math.sqrt(query.size(-1))
        
        # this only takes care of QK^T, bias must be added manually.
        logits = masked_matmul(query, key.transpose(-2, -1), mask)

        # compute angular part, this gives q dot (v' + v), need to subtract q dot v and normalize by dist
        angular = masked_matmul(query_vec, dst_vec.transpose(-2, -1), mask)
        angular = angular.values() - (query_vec * pos.expand(self.num_heads, -1, -1)).sum(-1)[:, row_index]

        logits = _wrap_value(logits, logits.values() + mask.values() + angular / dist)

        # Softmax to get the attention probabilities
        att = F.softmax(logits, dim=-1)

        # Optional dropout, could be part of the masking in the future
        att = self.att_drop(att)

        # normalize att
        norm_att = _wrap_value(att, att.values() / dist)

        # Get to the predicted values, for all heads
        dst_vec = torch.bmm(norm_att, dst_vec)
        avg_inv_dist = torch.bmm(
            norm_att, torch.ones((self.num_heads, col_pos.size(0), 1), device=col_pos.device)
        )
        src_vec = avg_inv_dist * pos.expand(self.num_heads, -1, -1)

        # subtract positions to get displacements, normalize it as well
        feat = F.normalize(dst_vec - src_vec, dim=-1)

        # get normal attention values
        y = torch.bmm(att, value)

        # combine them
        feat = torch.cat([y, feat, avg_inv_dist], dim=-1)

        # combine batched dimensions
        feat = feat.permute(1, 0, 2).reshape(x.size(0), -1)

        return self.output(feat)