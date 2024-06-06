# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn
import torch.nn.functional as F

from xformers.components.attention import Attention

from xformers.sparse import SparseCSRTensor
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

class SparseScaledDotProduct(Attention):

    def __init__(
        self,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attn_drop = nn.Dropout(dropout)

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
        att = masked_matmul(q, k.transpose(-2, -1), mask)
        att += mask

        # Softmax to get the attention probabilities
        att = F.softmax(att, dim=-1)

        #  Optional dropout, could be part of the masking in the future
        att = _apply_dropout(att, self.attn_drop)

        # Get to the predicted values, for all heads
        # y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
        y = torch.bmm(att, v)

        return y
