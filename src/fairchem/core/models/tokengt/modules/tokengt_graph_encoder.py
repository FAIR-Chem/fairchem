"""
Modified from https://github.com/microsoft/Graphormer
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .performer_pytorch import ProjectionUpdater
from .multihead_attention import MultiheadAttention
from .tokenizer import GraphFeatureTokenizer
from .tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer


def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class TokenGTGraphEncoder(nn.Module):
    def __init__(
            self,
            num_atoms: int,
            num_in_degree: int,
            num_out_degree: int,
            num_edges: int,
            num_spatial: int,
            num_edge_dis: int,
            edge_type: str,
            multi_hop_max_dist: int,

            rand_node_id: bool = False,
            rand_node_id_dim: int = 64,
            orf_node_id: bool = False,
            orf_node_id_dim: int = 64,
            lap_node_id: bool = False,
            lap_node_id_k: int = 8,
            lap_node_id_sign_flip: bool = False,
            lap_node_id_eig_dropout: float = 0.0,
            type_id: bool = False,

            stochastic_depth: bool = False,

            performer: bool = False,
            performer_finetune: bool = False,
            performer_nb_features: int = None,
            performer_feature_redraw_interval: int = 1000,
            performer_generalized_attention: bool = False,
            performer_auto_check_redraw: bool = True,

            num_encoder_layers: int = 12,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            encoder_normalize_before: bool = False,
            layernorm_style: str = "postnorm",
            apply_graphormer_init: bool = False,
            activation_fn: str = "gelu",
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,

            return_attention: bool = False

    ) -> None:

        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable
        self.performer = performer
        self.performer_finetune = performer_finetune

        self.graph_feature = GraphFeatureTokenizer(
            num_atoms=num_atoms,
            num_edges=num_edges,
            rand_node_id=rand_node_id,
            rand_node_id_dim=rand_node_id_dim,
            orf_node_id=orf_node_id,
            orf_node_id_dim=orf_node_id_dim,
            lap_node_id=lap_node_id,
            lap_node_id_k=lap_node_id_k,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            type_id=type_id,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers
        )
        self.performer_finetune = performer_finetune
        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if layernorm_style == "prenorm":
            self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        if stochastic_depth:
            assert layernorm_style == 'prenorm'  # only for residual nets

        self.cached_performer_options = None
        if self.performer_finetune:
            assert self.performer
            self.cached_performer_options = (
                performer_nb_features,
                performer_generalized_attention,
                performer_auto_check_redraw,
                performer_feature_redraw_interval
            )
            self.performer = False
            performer = False
            performer_nb_features = None
            performer_generalized_attention = False
            performer_auto_check_redraw = False
            performer_feature_redraw_interval = None

        self.layers.extend(
            [
                self.build_tokengt_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    encoder_layers=num_encoder_layers,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    drop_path=(0.1 * (layer_idx + 1) / num_encoder_layers) if stochastic_depth else 0,
                    performer=performer,
                    performer_nb_features=performer_nb_features,
                    performer_generalized_attention=performer_generalized_attention,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    layernorm_style=layernorm_style,
                    return_attention=return_attention,
                )
                for layer_idx in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

        if performer:
            # keeping track of when to redraw projections for all attention layers
            self.performer_auto_check_redraw = performer_auto_check_redraw
            self.performer_proj_updater = ProjectionUpdater(self.layers, performer_feature_redraw_interval)

    def performer_fix_projection_matrices_(self):
        self.performer_proj_updater.feature_redraw_interval = None

    def performer_finetune_setup(self):
        assert self.performer_finetune
        (
            performer_nb_features,
            performer_generalized_attention,
            performer_auto_check_redraw,
            performer_feature_redraw_interval
        ) = self.cached_performer_options

        for layer in self.layers:
            layer.performer_finetune_setup(performer_nb_features, performer_generalized_attention)

        self.performer = True
        self.performer_auto_check_redraw = performer_auto_check_redraw
        self.performer_proj_updater = ProjectionUpdater(self.layers, performer_feature_redraw_interval)

    def build_tokengt_graph_encoder_layer(
            self,
            embedding_dim,
            ffn_embedding_dim,
            encoder_layers,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            drop_path,
            performer,
            performer_nb_features,
            performer_generalized_attention,
            activation_fn,
            export,
            q_noise,
            qn_block_size,
            layernorm_style,
            return_attention,
    ):
        return TokenGTGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            encoder_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            drop_path=drop_path,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            layernorm_style=layernorm_style,
            return_attention=return_attention
        )

    def forward(
            self,
            batched_data,
            perturb=None,
            last_state_only: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        is_tpu = False

        if self.performer and self.performer_auto_check_redraw:
            self.performer_proj_updater.redraw_projections()

        if token_embeddings is not None:
            raise NotImplementedError
        else:
            x, padding_mask, padded_index = self.graph_feature(batched_data, perturb)

        # x: B x T x C

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        if attn_mask is not None:
            raise NotImplementedError

        attn_dict = {'maps': {}, 'padded_index': padded_index}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, attn = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=None)
            if not last_state_only:
                inner_states.append(x)
            attn_dict['maps'][i] = attn

        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), graph_rep, attn_dict
        else:
            return inner_states, graph_rep, attn_dict
