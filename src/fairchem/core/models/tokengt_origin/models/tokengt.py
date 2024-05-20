"""
Modified from https://github.com/jw9730/tokengt
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import (
    LayerNorm,
)

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel

from torch_scatter import scatter

from ..modules import init_graphormer_params, TokenGTGraphEncoder

logger = logging.getLogger(__name__)

@registry.register_model("tokengt_origin")
class TokenGTModel(BaseModel):
    """
    TokenGT, Tokenized Graph Transformer

    Parameters
    ----------
    num_atoms: int
        number of atom type
    use_pbc: bool
        use periodic boundary condition
    regress_forces: bool
        regress forces
    otf_graph: bool
        build graph on-the-fly
    max_neighbors: int
        maximum number of neighbors
    max_radius: float
        maximum radius
    max_num_elements: int
        maximum number of elements
    attention_dropout: float
        dropout prob for attention weights
    dropout: float
        dropout prob
    encoder_ffn_embed_dim: int
        encoder embedding dim for FFN
    encoder_layers: int
        num encoder layers
    encoder_attention_heads: int
        num encoder attention heads
    encoder_embed_dim: int
        encoder embedding dimension
    share_encoder_input_output_embed: bool
        share encoder input and output embeddings
    rand_node_id: bool
        use random feature node identifiers
    rand_node_id_dim: int
        dim of random node identifiers
    orf_node_id: bool
        use orthogonal random feature node identifiers
    orf_node_id_dim: int
        dim of orthogonal random node identifier
    lap_node_id: bool
        use Laplacian eigenvector node identifiers
    lap_node_id_dim: int
        number of Laplacian eigenvectors to use, from smallest eigenvalues
    lap_node_id_sign_flip: bool
        randomly flip the signs of eigvecs
    lap_node_id_eig_dropout: float
        dropout prob for Lap eigvecs
    type_id: bool
        use type identifiers
    stochastic_depth: bool
        use stochastic depth regularizer
    performer: bool
        linearized self-attention with Performer kernel
    performer_nb_features: int
        number of random features for Performer, defaults to (d*log(d)) where d is head dim
    performer_feature_redraw_interval: int
        how frequently to redraw the projection matrix for Performer
    performer_generalized_attention: bool
        defaults to softmax approximation, but can be set to True for generalized attention
    performer_finetune: bool
        load softmax checkpoint and fine-tune with performer
    apply_graphormer_init: bool
        use Graphormer initialization
    activation_fn: str
        activation to use
    encoder_normalize_before: bool
        apply layernorm before encoder
    prenorm: bool
        apply layernorm before self-attention and ffn
    postnorm: bool
        apply layernorm after self-attention and ffn
    """
    def __init__(
            self,
            num_atoms: int,  # not used
            bond_feat_dim: int,  # not used
            num_targets: int,  # not used
            num_elements: int = 100,
            use_pbc: bool = True,
            regress_forces: bool = True,
            otf_graph: bool = True,
            max_neighbors: int = 500,
            max_radius: float = 5.0,
            act_dropout: float =  0.0,
            attention_dropout: float = 0.1,
            dropout: float = 0.1,
            encoder_ffn_embed_dim: int = 4096,
            encoder_layers: int = 6,
            encoder_attention_heads: int = 8,
            encoder_embed_dim: int = 1024,
            share_encoder_input_output_embed: bool = False,
            rand_node_id: bool = False,
            rand_node_id_dim: int | None = None,
            orf_node_id: bool = False,
            orf_node_id_dim: int | None = None,
            lap_node_id: bool = False,
            lap_node_id_dim: int | None = None,
            lap_node_id_sign_flip: bool = False,
            lap_node_id_eig_dropout: float | None = None,
            type_id: bool = True,
            stochastic_depth: bool = False,
            performer: bool = False,
            performer_nb_features: int | None = None,
            performer_feature_redraw_interval: int | None = None,
            performer_generalized_attention: bool | None = None,
            performer_finetune: bool = False,
            apply_graphormer_init: bool = True,
            activation_fn: str = "gelu",
            encoder_normalize_before: bool = True,
            prenorm: bool = False,
            postnorm: bool = False,
        ):
        super().__init__()

        if apply_graphormer_init:
            self.apply(init_graphormer_params)

        self.encoder_embed_dim = encoder_embed_dim
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.regress_forces = regress_forces
        self.max_neighbors = max_neighbors
        self.cutoff = max_radius
        self.lap_node_id_dim = lap_node_id_dim

        self.encoder = TokenGTEncoder(
            num_elements=num_elements,
            regress_forces=regress_forces,
            act_dropout=act_dropout,
            attention_dropout=attention_dropout,
            dropout=dropout,
            encoder_ffn_embed_dim=encoder_ffn_embed_dim,
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_embed_dim=encoder_embed_dim,
            share_encoder_input_output_embed=share_encoder_input_output_embed,
            rand_node_id=rand_node_id,
            rand_node_id_dim=rand_node_id_dim,
            orf_node_id=orf_node_id,
            orf_node_id_dim=orf_node_id_dim,
            lap_node_id=lap_node_id,
            lap_node_id_dim=lap_node_id_dim,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            type_id=type_id,
            stochastic_depth=stochastic_depth,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_feature_redraw_interval=performer_feature_redraw_interval,
            performer_generalized_attention=performer_generalized_attention,
            performer_finetune=performer_finetune,
            apply_graphormer_init=apply_graphormer_init,
            activation_fn=activation_fn,
            encoder_normalize_before=encoder_normalize_before,
            prenorm=prenorm,
            postnorm=postnorm,
        )
        if performer_finetune:
            self.encoder.performer_finetune_setup()

    def forward(self, data):
        # OTF graph construction
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            *_ # unused outputs
        ) = self.generate_graph(
            data,
        )
        if self.lap_node_id_dim > 0:
            lap_vec = self.calc_lap(
                data, edge_index, self.lap_node_id_dim
            )
        else:
            lap_vec = None
        # extract data
        pos = data.pos
        batch = data.batch
        natoms = data.natoms
        edge_data = data.tags[edge_index[0]] * 3 + data.tags[edge_index[1]]
        atomic_numbers = data.atomic_numbers.long()

        energy, forces = self.encoder(
            batch,
            pos, 
            natoms,
            atomic_numbers, 
            edge_index,
            edge_distance_vec, 
            edge_distance,
            edge_data,
            lap_vec,
        )
        if self.regress_forces:
            return {"energy": energy, "forces": forces}
        else:
            return {"energy": energy}
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class TokenGTEncoder(nn.Module):
    def __init__(
            self,
            num_elements: int,
            regress_forces: bool,
            act_dropout: float =  0.0,
            attention_dropout: float = 0.1,
            dropout: float = 0.1,
            encoder_ffn_embed_dim: int = 4096,
            encoder_layers: int = 6,
            encoder_attention_heads: int = 8,
            encoder_embed_dim: int = 1024,
            share_encoder_input_output_embed: bool = False,
            rand_node_id: bool = False,
            rand_node_id_dim: int | None = None,
            orf_node_id: bool = False,
            orf_node_id_dim: int | None = None,
            lap_node_id: bool = False,
            lap_node_id_dim: int | None = None,
            lap_node_id_sign_flip: bool = False,
            lap_node_id_eig_dropout: float | None = None,
            type_id: bool = True,
            stochastic_depth: bool = False,
            performer: bool = False,
            performer_nb_features: int | None = None,
            performer_feature_redraw_interval: int | None = None,
            performer_generalized_attention: bool | None = None,
            performer_finetune: bool = False,
            apply_graphormer_init: bool = True,
            activation_fn: str = "gelu",
            encoder_normalize_before: bool = True,
            prenorm: bool = False,
            postnorm: bool = False,
        ):
        super().__init__()
        assert (prenorm != postnorm)
        self.encoder_layers = encoder_layers
        self.num_attention_heads = encoder_attention_heads

        if prenorm:
            layernorm_style = "prenorm"
        elif postnorm:
            layernorm_style = "postnorm"
        else:
            raise NotImplementedError

        self.graph_encoder = TokenGTGraphEncoder(
            # <
            num_elements=num_elements,
            # >
            # < for tokenization
            rand_node_id=rand_node_id,
            rand_node_id_dim=rand_node_id_dim,
            orf_node_id=orf_node_id,
            orf_node_id_dim=orf_node_id_dim,
            lap_node_id=lap_node_id,
            lap_node_id_dim=lap_node_id_dim,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            type_id=type_id,
            # >
            # <
            stochastic_depth=stochastic_depth,
            performer=performer,
            performer_finetune=performer_finetune,
            performer_nb_features=performer_nb_features,
            performer_feature_redraw_interval=performer_feature_redraw_interval,
            performer_generalized_attention=performer_generalized_attention,
            num_encoder_layers=encoder_layers,
            embedding_dim=encoder_embed_dim,
            ffn_embedding_dim=encoder_ffn_embed_dim,
            num_attention_heads=encoder_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=act_dropout,
            encoder_normalize_before=encoder_normalize_before,
            layernorm_style=layernorm_style,
            apply_graphormer_init=apply_graphormer_init,
            activation_fn=activation_fn,
            # >
        )

        self.share_input_output_embed = share_encoder_input_output_embed
        self.energy_out = nn.Sequential(
            LayerNorm(encoder_embed_dim),
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
            nn.GELU(),
            nn.Linear(encoder_embed_dim, 1)
        )
        self.forces_out =  nn.Sequential(
            LayerNorm(encoder_embed_dim),
            nn.Linear(encoder_embed_dim, encoder_embed_dim),
            nn.GELU(),
            nn.Linear(encoder_embed_dim, 3)
        )

        self.regress_forces = regress_forces

    def forward(
            self, 
            batch,
            pos, 
            natoms,
            atomic_numbers, 
            edge_index,
            edge_distance_vec, 
            edge_distance,
            edge_data,
            lap_vec,
        ):

        x, padded_node_mask = self.graph_encoder(
            batch,
            pos, 
            natoms,
            atomic_numbers, 
            edge_index,
            edge_distance_vec,
            edge_distance,
            edge_data,
            lap_vec,
        )
        x = x.transpose(0, 1)
        # project back to output sizes
        energy = self.energy_out(x[:, 0])
        if self.regress_forces:
            nodes = x[:, 2:][padded_node_mask]
            edge_inputs = torch.cat([nodes[edge_index[0]], nodes[edge_index[1]]], dim = 1)
            force_magnitudes = self.forces_out(edge_inputs)
            forces = scatter(
                src=force_magnitudes * edge_distance_vec / edge_distance[:, None], # could be improved by including 1/r potential ??
                index=edge_index[1],
                dim=0,
                dim_size=len(batch),
                reduce="sum"
            )
        else:
            forces = None
        return energy, forces

    def performer_finetune_setup(self):
        self.graph_encoder.performer_finetune_setup()
