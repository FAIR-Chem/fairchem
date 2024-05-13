"""
Modified from https://github.com/microsoft/Graphormer
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout

from .multihead_attention import MultiheadAttention
from .multihead_performer_attention import MultiheadPerformerAttention
from .feedforward import FeedForward
from .droppath import DropPath


class TokenGTGraphEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            encoder_layers: int = 12,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            drop_path: float = 0.0,
            performer: bool = False,
            performer_nb_features: int = None,
            performer_generalized_attention: bool = False,
            activation_fn: str = "relu",
            export: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
            init_fn: Callable = None,
            layernorm_style: str = "postnorm",
            return_attention: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        self.layernorm_style = layernorm_style
        self.return_attention = return_attention

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            attention_dropout=attention_dropout,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # drop path for stochastic depth
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.feedforward = self.build_FFN(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise,
            qn_block_size,
            activation_fn,
            activation_dropout,
            dropout,
            module_name=self.__class__.__name__
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # drop path for stochastic depth
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def build_FFN(
            self,
            embedding_dim,
            ffn_embedding_dim,
            q_noise,
            qn_block_size,
            activation_fn,
            activation_dropout,
            dropout,
            module_name
    ):
        return FeedForward(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            activation_fn=activation_fn,
            activation_dropout=activation_dropout,
            dropout=dropout,
            module_name=module_name,
        )

    def build_self_attention(
            self,
            embed_dim,
            num_attention_heads,
            performer,
            performer_nb_features,
            performer_generalized_attention,
            attention_dropout,
            dropout,
            self_attention,
            q_noise,
            qn_block_size
    ):
        if performer:
            return MultiheadPerformerAttention(
                embed_dim,
                num_attention_heads,
                performer_nb_features=performer_nb_features,
                performer_generalized_attention=performer_generalized_attention,
                attention_dropout=attention_dropout,
                dropout=dropout,
                self_attention=True,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )
        else:
            return MultiheadAttention(
                embed_dim,
                num_attention_heads,
                attention_dropout=attention_dropout,
                dropout=dropout,
                self_attention=True,
                q_noise=q_noise,
                qn_block_size=qn_block_size
            )

    def performer_finetune_setup(self, performer_nb_features, performer_generalized_attention):
        self.self_attn.performer_finetune_setup(performer_nb_features, performer_generalized_attention)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_bias: Optional[torch.Tensor] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        if self.layernorm_style == "prenorm":
            residual = x
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                attn_bias=self_attn_bias,
                key_padding_mask=self_attn_padding_mask,
                need_weights=self.return_attention,
                need_head_weights=self.return_attention,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = self.drop_path1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.feedforward(x)
            x = self.drop_path2(x)
            x = residual + x

        elif self.layernorm_style == "postnorm":
            residual = x
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                attn_bias=self_attn_bias,
                key_padding_mask=self_attn_padding_mask,
                need_weights=self.return_attention,
                need_head_weights=self.return_attention,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.feedforward(x)
            x = residual + x
            x = self.final_layer_norm(x)

        else:
            raise NotImplementedError
        return x, attn
