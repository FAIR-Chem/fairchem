"""
Author: Ryan Liu
"""
import logging
import math
from typing import List, Union
from random import random

import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel

from torch_scatter import scatter

from .encoder_layer import EncoderLayer
from .pair_embed import PairEmbed
from .mlp import ResMLP
from .pbc_utils import build_radius_graph

logger = logging.getLogger(__name__)

@registry.register_model("transformer")
class Transformer(BaseModel):
    """
    Transformer

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
    att_dropout: float
        attention dropout rate
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
            att_dropout: float = 0.,
            num_layers: int = 12,
            num_heads: int = 8,
            otf_graph: bool = True,
            rbf_radius: float = 5.0,
            use_pbc: bool = False,
            num_gaussians: int = 50,
            output_layers: int = 3,
            avg_atoms: float = 60,
            stochastic_depth: float = 0.,
        ):

        super().__init__()

        self.otf_graph = otf_graph
        self.rbf_radius = rbf_radius
        self.use_pbc = use_pbc
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_layers = output_layers
        self.avg_atoms = avg_atoms
        self.stochastic_depth = stochastic_depth

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
            num_elements=len(elements),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_masks=2*num_layers,
            num_gaussians=num_gaussians,
            rbf_radius=rbf_radius,
            dropout=dropout,
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                att_dropout=att_dropout,
            ) for _ in range(num_layers)
        ])
        
        self.forces_out = ResMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=3,
            num_layers=output_layers,
            bias_output=False,
        )

        self.energy_out = ResMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=output_layers,
            bias_output=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(
            self.forces_out.input.weight,
            -math.sqrt(2 / (self.embed_dim * self.num_layers)),
            math.sqrt(2 / (self.embed_dim * self.num_layers))
        )
        nn.init.uniform_(
            self.energy_out.input.weight,
            -math.sqrt(2 / (self.embed_dim * self.num_layers)),
            math.sqrt(2 / (self.embed_dim * self.num_layers))
        )
        nn.init.uniform_(
            self.energy_out.output.weight,
            -math.sqrt(1 / (self.hidden_dim * self.avg_atoms * (self.output_layers + 1))),
            math.sqrt(1 / (self.hidden_dim * self.avg_atoms * (self.output_layers + 1)))
        )

    def forward(self, data):

        # extract data
        batch = data.batch
        pos = data.pos
        atomic_numbers = self.atomic_number_mask[data.atomic_numbers.long()]

        # build graph on-the-fly
        row_index, col_index, dist, col_pos, to_col_index = build_radius_graph(data, self.rbf_radius, self.use_pbc)
        
        # initialize inputs
        x = self.atomic_number_encoder(atomic_numbers)

        # get pair embeddings
        att_bias = self.pair_embed(atomic_numbers, row_index, col_index, to_col_index, dist)
        att_bias, pos_att_bias = att_bias[:self.num_layers], att_bias[self.num_layers:]

        # forward passing
        for i in range(self.num_layers):
            if self.training and random() < self.stochastic_depth:
                continue
            # encoder block
            x = self.layers[i](
                x,
                row_index,
                col_index,
                to_col_index,
                att_bias[i],
                pos_att_bias[i],
                dist,
                pos,
                col_pos,
            )
            
        # get outputs
        energy = self.energy_out(x)
        forces = self.forces_out(x)

        # averge over all energies
        energy = scatter(
            energy, batch, dim=0, reduce="sum"
        )

        return {"energy": energy, "forces": forces}
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @torch.jit.ignore
    def no_weight_decay(self):
        # no weight decay on layer norms and embeddings
        # ref: https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        no_wd_list.append("forces_out.output.weight")
        no_wd_list.append("energy_out.output.weight")
        return set(no_wd_list)
