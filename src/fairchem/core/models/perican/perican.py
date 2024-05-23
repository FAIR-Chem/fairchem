"""
Author: Ryan Liu
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel

from .layers import T2toT2, OutputModule
from .features import FeatureBuilder

logger = logging.getLogger(__name__)

@registry.register_model("PERICAN")
class PERICAN(BaseModel):
    """
    Permutation Equivariant Rotation Invariant/Covariant Aggregator Network

    Parameters
    ----------
    otf_graph: bool
        to build the graph on-the-fly
    embed_dim: int
        embedding dimention
    eq_dim: int
        number of channels to aggregate
    num_element: int
        maximum number of elements
    rbf_radius: float
        the radius cutoff to use in RBFs
    num_gaussians: int
        number of gaussians in RBFs
    avg_atoms: float
        average numnber of atoms
    dropout: float 
        dropout rate
    agg_fns: List[str]
        aggregators
    num_layers: int
        number of T2->T2 layers
    output_layers: int
        number of output layers
    """
    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        otf_graph: bool = True,
        embed_dim: int = 128,
        eq_dim: int = 128,
        num_element: int = 100,
        rbf_radius: float = 12.,
        num_gaussians: int = 50,
        avg_atoms: float = 60.,
        dropout: float = 0.,
        agg_fns: List[str] = ["sum"],
        num_layers: int = 12,
        output_layers: int = 4,
    ):
        super().__init__()

        self.otf_graph = otf_graph
        self.num_layers = num_layers

        self.feature_builder = FeatureBuilder(
            embed_dim=embed_dim,
            num_element=num_element,
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians,       
        )

        self.layers = nn.ModuleList([
            T2toT2(
                in_channels=embed_dim,
                eq_channels=eq_dim,
                out_channels=embed_dim,
                avg_len=avg_atoms,
                dropout=dropout,
                agg_fns=agg_fns,
            ) for _ in range(num_layers)
        ])

        self.output_modules = nn.ModuleList([
            OutputModule(
                embed_dim=embed_dim,
                rbf_radius=rbf_radius,
                num_gaussians=num_gaussians,
                num_layers=output_layers,
                dropout=dropout,
                avg_atoms=avg_atoms
            ) for _ in range(num_layers + 1)
        ])

    def forward(self, data) -> Dict[str, torch.Tensor]:
        pos = data.pos
        natoms = data.natoms
        atomic_numbers = data.atomic_numbers.long()
        
        T, mask, node_mask, dist, vec_hat = self.feature_builder(
            pos,
            natoms,
            atomic_numbers,
        )

        energy, forces = self.output_modules[0](
            T, mask, dist, vec_hat
        )

        for i, layer in enumerate(self.layers):
            T = T + layer(T, mask)
            e, f = self.output_modules[i+1](
                T, mask, dist, vec_hat
            )
            energy += e
            forces += f

        forces = forces.transpose(0, 1)[node_mask]

        energy = energy / self.num_layers
        forces = forces / self.num_layers

        return {"energy": energy, "forces": forces} 
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())