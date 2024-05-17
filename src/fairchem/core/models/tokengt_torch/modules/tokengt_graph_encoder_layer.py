"""
Modified from https://github.com/jw9730/tokengt
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from .feedforward import FeedForward


class TokenGTGraphEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            encoder_layers: int = 12,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_fn: str = "relu",
            layernorm_style: str = "postnorm",
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.encoder_layers = encoder_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.layernorm_style = layernorm_style

        # Initialize blocks
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.feedforward = self.build_FFN(
            self.embedding_dim,
            ffn_embedding_dim,
            activation_fn,
            dropout,
        )

        self.dropout_module = nn.Dropout(dropout)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def build_FFN(
            self,
            embedding_dim,
            ffn_embedding_dim,
            activation_fn,
            dropout,
    ):
        return FeedForward(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            activation_fn=activation_fn,
            dropout=dropout,
        )

    def forward(
            self,
            x: torch.Tensor,
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
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.feedforward(x)
            x = residual + x

        elif self.layernorm_style == "postnorm":
            residual = x
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
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
        return x
