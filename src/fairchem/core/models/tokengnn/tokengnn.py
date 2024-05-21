"""
Author: Ryan Liu
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel

from .encoder_layers import TokenGNNEncoderLayer, OutputModule
from .tokenizer import GraphFeatureTokenizer

logger = logging.getLogger(__name__)

@registry.register_model("TokenGNN")
class TokenGNN(BaseModel):
    """
    Tokenized Graph Neural Network

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
        max_neighbors: int = 50,
        max_radius: float = 5.0,
        dropout: float = 0.1,
        num_gaussians: int = 50,
        output_layers: int = 3,
    ):

        super().__init__()
        
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.cutoff = max_radius
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TokenGNNEncoderLayer(
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                num_heads=num_heads,
                max_radius=max_radius,
                num_gaussians=num_gaussians
            ) for _ in range(num_layers)
        ])

        self.output_modules = nn.ModuleList([
            OutputModule(
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                hidden_layers=output_layers,
                dropout=dropout,
                num_gaussians=num_gaussians,
                max_radius=max_radius
            ) for _ in range(num_layers + 1)
        ])

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
        

    def forward(self, data):

        # prepare inputs
        batch = data.batch.long()
        natoms = data.natoms.long()
        pos = data.pos
        atomic_numbers = data.atomic_numbers.long()

        # OTF graph construction
        (
            edge_index,
            *_ # unused outputs
        ) = self.generate_graph(data)

        # symmetrize the graph
        order = edge_index[0] >  edge_index[1]
        edge_index[:, order] = edge_index[:, order].flip(0)
        edge_index = torch.unique(edge_index, dim=1)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # tokenization
        (
            x, edges, vec_hat, all_dist, dist ,cosine_dd, cosine_ss
        ) = self.tokenizer(pos, natoms, atomic_numbers, edge_index)
        
        # first pass
        energy, forces = self.output_modules[0](
            x,
            pos,
            dist,
            vec_hat,
            batch,
            edge_index,
        )

        # encoder layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edges, all_dist, dist, cosine_dd, cosine_ss)
            e, f = self.output_modules[i+1](
                x,
                pos,
                dist,
                vec_hat,
                batch,
                edge_index,
            )
            energy += e
            forces += f
        
        energy = self.energy_scale * energy
        forces = self.force_scale * forces

        return {"energy": energy, "forces": forces} 
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())