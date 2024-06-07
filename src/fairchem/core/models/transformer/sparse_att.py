import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from xformers.components.attention import Attention
from xformers.sparse import SparseCSRTensor
from xformers.sparse.utils import _coo_to_csr
from xformers.ops import masked_matmul

def _apply_dropout(
        att: SparseCSRTensor,
        dropout: nn.Module
    ):
    values = att.values().clone()
    values = dropout(values)
    att = SparseCSRTensor._wrap(
        att.shape,
        values,
        att._csr_row_indices,
        att._csr_row_offsets,
        att._csr_column_indices,
        att._csr_transp_info,
    )
    return att

def _from_coo(m, n, rows, cols, vals):
    rows, cols = rows.int(), cols.int()
    assert torch.unique(torch.stack([rows, cols]), dim=1).size(1) == len(cols), "coo must be coaleased"
    indx = torch.argsort(rows)
    rows, cols, vals = rows[indx], cols[indx], vals[indx]

    if len(vals) % 4 != 0:
        # remove the four smallest item
        # only make sense for additive attention mask
        # if used for other sparse operation consider modify this!
        mask = torch.argsort(vals.amax(-1))[len(vals)%4:]
        rows, cols, vals = rows[mask], cols[mask], vals[mask]

    row_offsets, column_indices = _coo_to_csr(m, n, rows, cols)
    return SparseCSRTensor(row_offsets, column_indices, vals.T, (vals.size(1), m, n))

class SparseScaledDotProduct(Attention):

    def __init__(
        self,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.att_drop = nn.Dropout(dropout)

        # Properties specific to this attention mechanism
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: SparseCSRTensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        mask must be a 3D additive SparseCSRTensor mask.
        since batch dimension must have same sparse pattern, it can only be used
        for head. The remaining should use implicit batching.
        """

        # Attend: (B x nh, S, hs) x (B x nh, hs, S) -> (B x nh, S, S)
        q = q / math.sqrt(k.size(-1))
        
        # this only takes care of QK^T, bias must be added manually.
        logits = masked_matmul(q, k.transpose(-2, -1), mask)
        logits = logits + mask

        # Softmax to get the attention probabilities
        att = F.softmax(logits, dim=-1)

        #  Optional dropout, could be part of the masking in the future
        att = _apply_dropout(att, self.att_drop)

        # Get to the predicted values, for all heads
        # y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
        y = torch.bmm(att, v)

        return y, logits

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
    ):
        output = self.linear(x)
        output = output.reshape(x.size(0), self.num_heads, self.d_k)
        output = output.permute(1, 0, 2)

        return output
    
class SparseSelfAttention(nn.Module):
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

        self.attention = SparseScaledDotProduct(dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        att_bias: Optional[torch.Tensor] = None,
        need_weights: Optional[bool] = False
    ):
        """
        Current implementation requires implicit batching!
        """
        # project to qkv
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # construct CSR format mask
        mask = _from_coo(x.size(0), x.size(0), edge_index[0], edge_index[1], att_bias)

        # compute scaled dot product attention
        self_att, logits = self.attention(query, key, value, mask)
        self_att = self_att.permute(1, 0, 2).reshape(x.size(0), -1)

        # output projection
        self_att = self.output(self_att)

        if need_weights:
            return self_att, logits
        else:
            return self_att