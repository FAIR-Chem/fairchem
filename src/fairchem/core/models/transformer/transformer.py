"""
Author: Ryan Liu
"""
import logging

import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel

from .layers import SelfAttentionLayer, PairEmbed, OutputModule



logger = logging.getLogger(__name__)

@registry.register_model("transformer")
class Transformer(BaseModel):
    """
    Pure Transformer

    Parameters
    ----------
    num_elements: int
        the number of possible elements
    embed_dim: int
        the dimensionality of embedding used
    ff_dim: int
        the hidde channels of feed forward
    num_layers: int
        the number of layers to use
    dropout: float
        dropout rate
    heads: int
        number of attention heads
    otf_graph: bool
        calculate graph on-the-fly
    rbf_radius: float
        rbf radius
    num_gaussians: int
        number of gaussians to use in rbf
    avg_atoms: float
        average number of atoms
    """
    def __init__(
            self,
            num_atoms: int,  # not used
            bond_feat_dim: int,  # not used
            num_targets: int,  # not used
            num_elements: int = 100,
            embed_dim: int = 512,
            ff_dim: int = 1024,
            num_layers: int = 12,
            dropout: float = 0.,
            heads: int = 8,
            otf_graph: bool = False,
            rbf_radius: float = 5.0,
            num_gaussians: int = 50,
            output_layers: int = 3,
            avg_atoms: float = 60,
        ):

        super().__init__()

        self.otf_graph=otf_graph
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

        self.atomic_number_encoder = nn.Embedding(
            num_elements,
            embed_dim,
            padding_idx=0
        )

        self.pair_embedding = PairEmbed(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            num_heads=heads,
            num_layers=num_layers,
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians
        )

        self.layers = nn.ModuleList([
            SelfAttentionLayer(
                embed_dim,
                ff_dim,
                heads,
                dropout,
            ) for _ in range(num_layers)
        ])

        self.output_modules = nn.ModuleList([
            OutputModule(
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                hidden_layers=output_layers,
                num_gaussians=num_gaussians,
                max_radius=rbf_radius
            ) for i in range(num_layers + 1)
        ])

        self.energy_scale = 1 / (num_layers * avg_atoms * avg_atoms)
        self.force_scale = 1 / (num_layers * avg_atoms)

    def forward(self, data):

        # extract data
        pos = data.pos
        natoms = data.natoms
        atomic_numbers = data.atomic_numbers.long()

        # tokenize the sequence
        nmax = natoms.max()
        batch_size = len(natoms)
        token_pos = torch.arange(nmax, device=natoms.device)[None, :].repeat(batch_size, 1)
        padded_node_mask = token_pos < natoms[:, None]

        padded_pos = torch.zeros((batch_size, nmax, 3), device=pos.device, dtype=torch.float)
        padded_pos[padded_node_mask] = pos
        padded_pos = padded_pos.transpose(0, 1).contiguous()

        padded_anum = torch.zeros((batch_size, nmax), device=natoms.device, dtype=torch.long)
        padded_anum[padded_node_mask] = atomic_numbers
        padded_anum = padded_anum.transpose(0, 1).contiguous()

        # encode the sequence
        x = self.pos_encoder(padded_pos) + self.atomic_number_encoder(padded_anum)
        key_padding_mask = torch.zeros_like(padded_node_mask, dtype=torch.float)
        key_padding_mask.masked_fill_(~padded_node_mask, -torch.inf)

        # pair embed
        attn_masks, pairs = self.pair_embedding(x, padded_pos, padded_node_mask)

        # initialize outputs
        energy, forces = self.output_modules[0](
            x,
            pairs,
            padded_pos,
            padded_node_mask,
        )

        # transformer layers
        for layer, output_layer, attn_mask in zip(self.layers, self.output_modules[1:], attn_masks):
            x = layer(x, key_padding_mask, attn_mask)
            E, F = output_layer(
                x,
                pairs,
                padded_pos,
                padded_node_mask,
            )
            energy += E
            forces += F

        energy = self.energy_scale * energy
        forces = self.force_scale * forces

        return {"energy": energy, "forces": forces}
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())