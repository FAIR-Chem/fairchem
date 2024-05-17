"""
Modified from https://github.com/jw9730/tokengt
"""

import torch
import torch.nn as nn

from .tokenizer import GraphFeatureTokenizer
from .tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer


class TokenGTGraphEncoder(nn.Module):
    def __init__(
            self,
            num_elements: int = 100,
            rand_node_id: bool = False,
            rand_node_id_dim: int = 64,
            orf_node_id: bool = False,
            orf_node_id_dim: int = 64,
            lap_node_id: bool = False,
            lap_node_id_dim: int = 8,
            lap_node_id_sign_flip: bool = False,
            lap_node_id_eig_dropout: float = 0.0,
            type_id: bool = False,

            num_encoder_layers: int = 12,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            encoder_normalize_before: bool = False,
            layernorm_style: str = "postnorm",
            activation_fn: str = "gelu",
    ) -> None:

        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

        self.graph_feature = GraphFeatureTokenizer(
            num_elements=num_elements,
            rand_node_id=rand_node_id,
            rand_node_id_dim=rand_node_id_dim,
            orf_node_id=orf_node_id,
            orf_node_id_dim=orf_node_id_dim,
            lap_node_id=lap_node_id,
            lap_node_id_dim=lap_node_id_dim,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            type_id=type_id,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers
        )

        if encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        if layernorm_style == "prenorm":
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_tokengt_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    encoder_layers=num_encoder_layers,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_fn=activation_fn,
                    layernorm_style=layernorm_style,
                )
                for _ in range(num_encoder_layers)
            ]
        )
    
    def build_tokengt_graph_encoder_layer(
            self,
            embedding_dim,
            ffn_embedding_dim,
            encoder_layers,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_fn,
            layernorm_style,
    ):
        return TokenGTGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            encoder_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_fn=activation_fn,
            layernorm_style=layernorm_style,
        )

    def forward(
            self,
            batch,
            pos, 
            natoms,
            atomic_numbers, 
            edge_index,
            edge_distance_vec,
            edge_distance,
            lap_vec,
        ):

        x, padding_mask, padded_node_mask = self.graph_feature(
            batch,
            pos, 
            natoms,
            atomic_numbers, 
            edge_index,
            edge_distance_vec,
            edge_distance,
            lap_vec,
        )

        # x: B x T x C

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x, self_attn_padding_mask=padding_mask)
        
        return x, padded_node_mask
