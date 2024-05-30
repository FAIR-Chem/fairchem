"""
Author: Ryan Liu
"""
import logging
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_cluster.radius import radius_graph

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel

from .attn import SelfAttentionLayer
from .output import OutputModule
from .pair_embed import PairEmbed

logger = logging.getLogger(__name__)

@registry.register_model("sparse_transformer")
class SparseTransformer(BaseModel):
    """
    Sparse Transformer

    Parameters
    ----------
    elements: List[int]
        list of possible atomic numbers
    embed_dim: int
        the dimensionality of embedding used
    hidden_dim: int
        the hidde channels of feed forward
    dropout: float
        dropout rate
    num_layers: int
        the number of layers to use
    num_heads: int
        number of attention heads
    otf_graph: bool
        calculate graph on-the-fly
    rbf_radius: float
        rbf radius
    use_pbc: bool
        to use periodic boundary condition or not
    max_neighbors: int
        maximum number of neighbors
    num_gaussians: int
        number of gaussians to use in rbf
    trainable_rbf: bool
        use trainable RBFs
    output_layers: int
        number of output layers to use
    avg_atoms: float
        average number of atoms
    """
    def __init__(
            self,
            num_atoms: int,  # not used
            bond_feat_dim: int,  # not used
            num_targets: int,  # not used
            elements: Union[int, List[int]] = 100,
            embed_dim: int = 128,
            hidden_dim: int = 128,
            dropout: float = 0.,
            num_layers: int = 12,
            num_heads: int = 8,
            otf_graph: bool = False,
            rbf_radius: float = 5.0,
            use_pbc: bool = False,
            max_neighbors: int = 32,
            num_gaussians: int = 50,
            num_pair_embed_layers: int = 2,
            output_layers: int = 3,
            avg_atoms: float = 60,
        ):

        super().__init__()

        self.otf_graph = otf_graph
        self.cutoff = rbf_radius
        self.max_neighbors = max_neighbors
        self.use_pbc = use_pbc
        self.num_layers = num_layers

        if isinstance(elements, int):
            self.register_buffer("atomic_number_mask", torch.arange(elements + 1))
        elif isinstance(elements, list):
            atomic_number_mask = - torch.ones(max(elements) + 1, dtype=torch.long)
            atomic_number_mask[elements] = torch.arange(len(elements))
            self.register_buffer("atomic_number_mask", atomic_number_mask)
        else:
            raise TypeError("elements must be integer or list of integers")

        self.atomic_number_encoder = nn.Embedding(
            len(elements),
            embed_dim,
        )

        self.pair_embed = PairEmbed(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_masks=num_layers,
            num_gaussians=num_gaussians,
            rbf_radius=rbf_radius,
            dropout=dropout,
            num_layers=num_pair_embed_layers,
        )

        self.layers = nn.ModuleList([
            SelfAttentionLayer(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])

        self.init_output = OutputModule(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            hidden_layers=output_layers,
            num_gaussians=num_gaussians,
            rbf_radius=rbf_radius,
            avg_len=avg_atoms,
            connectivity=max_neighbors,
        )

        self.outputs = nn.ModuleList([
            OutputModule(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                hidden_layers=output_layers,
                num_gaussians=num_gaussians,
                rbf_radius=rbf_radius,
                avg_len=avg_atoms,
                connectivity=max_neighbors,
            ) for _ in range(num_layers)
        ])        

    def forward(self, data):

        # extract data
        batch = data.batch
        atomic_numbers = self.atomic_number_mask[data.atomic_numbers.long()]

        # build graph on-the-fly
        (
            edge_index,
            dist,
            vec,
            *_ #unused
        ) = self.generate_graph(data, enforce_max_neighbors_strictly=False)
        vec_hat = F.normalize(vec, dim=-1)

        # initialize inputs
        x = self.atomic_number_encoder(atomic_numbers)

        # initialize output
        energy, forces = self.init_output(x, edge_index, batch, dist, vec_hat)

        # get pair embeddings
        attn_masks = self.pair_embed(x, edge_index, dist)

        # forward passing
        for i in range(self.num_layers):
            # attention block
            x = self.layers[i](x, edge_index, attn_masks[i])

            # residual outputs
            e, f = self.outputs[i](x, edge_index, batch, dist, vec_hat)
            energy += e
            forces += f

        energy = energy / (self.num_layers + 1)
        forces = forces / (self.num_layers + 1)

        return {"energy": energy, "forces": forces}
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())