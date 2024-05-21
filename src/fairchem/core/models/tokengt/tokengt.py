"""
Author: Ryan Liu
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel

from .encoder_layers import AttentionBias, TokenGTEncoderLayer, OutputModule
from .tokenizer import GraphFeatureTokenizer
from .utils import ResMLP

logger = logging.getLogger(__name__)

@registry.register_model("TokenGT")
class TokenGT(BaseModel):
    """
    Tokenized Transformer

    Parameters
    ----------
    num_elements: int
        the number of possible elements
    emb_dim: int
        the dimensionality of embedding used
    ff_dim: int
        the hidde channels of feed forward
    num_heads: int
        number of heads
    num_layers: int
        the number of layers to use
    avg_atoms: float
        averge number of atoms per graph
    otf_graph: bool
        calculate graph on-the-fly
    max_neighbors: int
        maximum number of neighbors in OTF graph
    max_radius: float
        max search radius
    dropout: float
        dropout rate
    lap_node_id_dim: int
        the dimensionality of laplacian search
    heads: int
        number of attention heads
    num_gaussians: int
        number of gaussians to use in rbf
    use_attn_mask: bool
        to use attention mask or not
    """
    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        num_elements: int = 100,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 12,
        avg_atoms: float = 60,
        use_pbc: bool = True,
        otf_graph: bool = True,
        max_neighbors: int = 500,
        max_radius: float = 5.0,
        dropout: float = 0.1,
        lap_node_id_dim: int = 0,
        num_gaussians: int = 50,
        use_attn_mask: bool = True,
        output_layers: int = 3,
    ):
        super().__init__()

        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.cutoff = max_radius
        self.lap_node_id_dim = lap_node_id_dim
        self.use_attn_mask = use_attn_mask
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TokenGTEncoderLayer(
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                num_heads=num_heads,
            ) for _ in range(num_layers)
        ])

        self.output_modules = nn.ModuleList([
            OutputModule(
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                hidden_layers=output_layers,
                num_gaussians=num_gaussians,
                max_radius=max_radius
            ) for _ in range(num_layers + 1)
        ])

        if use_attn_mask:
            self.attn_bias = AttentionBias(
                ff_dim=ff_dim,
                dropout=dropout,
                num_heads=num_heads,
                num_layers=num_layers
            )

        if lap_node_id_dim > 0:
            self.lap_emb = ResMLP(
                input_dim=lap_node_id_dim,
                hidden_dim=ff_dim,
                output_dim=embed_dim,
                dropout=dropout
            )

        self.tokenizer = GraphFeatureTokenizer(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            num_elements=num_elements,
            rbf_radius=max_radius,
            num_gaussians=num_gaussians
        )

        self.energy_scale = 1 / (num_layers * avg_atoms * max_neighbors)
        self.force_scale = 1 / (num_layers * max_neighbors)
        

    def forward(self, data) -> Dict[str, torch.Tensor]:
        # prepare inputs
        batch = data.batch.long()
        pos = data.pos
        natoms = data.natoms.long()
        atomic_numbers = data.atomic_numbers.long()
        # OTF graph construction
        (
            edge_index,
            *_ # unused outputs
        ) = self.generate_graph(data)

        # tokenization
        (
            padded_features,
            padded_mask,
            padded_node_mask,
            padded_edge_mask,
            padded_index
        ) = self.tokenizer(batch, pos, natoms, atomic_numbers, edge_index)

        key_padding_mask = torch.zeros_like(padded_mask, dtype=torch.float)
        key_padding_mask.masked_fill_(~padded_mask, -torch.inf)
        
        if self.lap_node_id_dim > 0:
            lap_vec = self.calc_lap(
                data, edge_index, self.lap_node_id_dim
            )
            padded_features[padded_node_mask] += self.lap_emb(lap_vec)

        if self.use_attn_mask:
            attn_mask = self.attn_bias(
                padded_mask, 
                padded_node_mask, 
                padded_index
            )
        else:
            attn_mask = [None] * self.num_layers

        # transpose
        x = padded_features.transpose(0, 1)
        
        # first pass
        energy, forces = self.output_modules[0](
            x,
            pos,
            batch,
            edge_index,
            padded_node_mask,
            padded_edge_mask
        )

        # encoder layers
        for i, layer in enumerate(self.layers):
            x = layer(x, key_padding_mask, attn_mask[i])
            e, f = self.output_modules[i+1](
                x,
                pos,
                batch,
                edge_index,
                padded_node_mask,
                padded_edge_mask
            )
            energy += e
            forces += f
        
        energy = self.energy_scale * energy
        forces = self.force_scale * forces

        return {"energy": energy, "forces": forces} 
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())