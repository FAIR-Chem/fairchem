from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .multihead_attention import MultiheadAttention
from .performer_pytorch import FastAttention


class MultiheadPerformerAttention(MultiheadAttention):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            performer_nb_features=None,
            performer_generalized_attention=False,
            attention_dropout=0.0,
            dropout=0.0,
            bias=True,
            self_attention=False,
            q_noise=0.0,
            qn_block_size=8,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            attention_dropout,
            dropout,
            bias,
            self_attention,
            q_noise,
            qn_block_size
        )
        assert attention_dropout == 0.0
        self.fast_attention = FastAttention(
            self.head_dim,
            performer_nb_features,
            causal=False,
            generalized_attention=performer_generalized_attention,
            kernel_fn=nn.ReLU(),
            no_projection=False
        )

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            attn_bias: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        return self.forward_performer(
            query,
            key,
            value,
            attn_bias,
            key_padding_mask,
            need_weights,
            attn_mask,
            before_softmax,
            need_head_weights
        )
