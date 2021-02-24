"""
---

This code borrows heavily from the SE(3)-Transformers implementation as part of: https://github.com/FabianFuchsML/se3-transformer-public. License:

---
Copyright (c) 2020 Fabian Fuchs and Daniel Worrall

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Dict, List, Tuple

import dgl
import torch
from torch import nn
from torch.nn import Embedding
from torch.nn import functional as F

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import get_pbc_distances, radius_graph_pbc
from ocpmodels.models.utils.se3_utils.fibers import Fiber
from ocpmodels.models.utils.se3_utils.from_se3cnn import utils_steerable
from ocpmodels.models.utils.se3_utils.modules import (
    GAvgPooling,
    GConvSE3,
    GMaxPooling,
    GNormSE3,
    GSE3Res,
    get_basis,
    get_basis_and_r,
)


@registry.register_model("tfn")
class TFN(nn.Module):
    """SE(3) equivariant GCN

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Unused argument
        num_layers (int): Number of equivariant layers
        atom_feature_size (int): Node feature embedding size
        num_channels (int): Number of channels per representation degree
        num_degrees (int): Degree  of representation
        edge_dim (int): Additional edge features
        div (int): Divisor for # channels in attention values
        pooling (str): Pooling scheme ("avg" or "max")
        n_heads (int): Number of attention layers

    """

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        num_layers: int,
        atom_feature_size: int,
        num_channels: int,
        num_nlayers: int = 1,
        num_degrees: int = 4,
        edge_dim: int = 0,
        use_pbc: bool = True,
        regress_forces: bool = True,
    ):

        super().__init__()
        # Build the network
        self.atom_feature_size = atom_feature_size
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels * num_degrees
        self.edge_dim = edge_dim
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces

        self.fibers = {
            "in": Fiber(1, atom_feature_size),
            "mid": Fiber(num_degrees, self.num_channels),
            "out": Fiber(1, self.num_channels_out),
        }

        self.embedding = Embedding(100, atom_feature_size)

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks
        print(self.block0)
        print(self.block1)
        print(self.block2)

    def _build_gcn(self, fibers, out_dim):

        block0 = []
        fin = fibers["in"]
        for i in range(self.num_layers - 1):
            block0.append(
                GConvSE3(
                    fin,
                    fibers["mid"],
                    self_interaction=True,
                    edge_dim=self.edge_dim,
                )
            )
            block0.append(GNormSE3(fibers["mid"], num_layers=self.num_nlayers))
            fin = fibers["mid"]
        block0.append(
            GConvSE3(
                fibers["mid"],
                fibers["out"],
                self_interaction=True,
                edge_dim=self.edge_dim,
            )
        )

        block1 = [GMaxPooling()]

        block2 = []
        block2.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        block2.append(nn.ReLU(inplace=True))
        block2.append(nn.Linear(self.num_channels_out, out_dim))

        return (
            nn.ModuleList(block0),
            nn.ModuleList(block1),
            nn.ModuleList(block2),
        )

    def forward(self, data):

        pos = data.pos
        if self.regress_forces:
            pos = pos.requires_grad_(True)

        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
                return_distance_vec=True,
            )

            data.edge_index = out["edge_index"]

        G = self.pyg2dgl(data, out["distance_vec"], use_pbc=self.use_pbc)

        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {"0": G.ndata["f"]}
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.block1:
            h = layer(h, G)

        for layer in self.block2:
            h = layer(h)

        if self.regress_forces:
            energy = h
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    pos,
                    grad_outputs=torch.ones_like(h),
                    create_graph=True,
                )[0]
            )
            return energy, forces

        else:
            return h

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def pyg2dgl(self, data, pbc_dist_vec, use_pbc=True):
        src, dst = data.edge_index
        G = dgl.DGLGraph((src.cpu(), dst.cpu()))

        G.ndata["x"] = data.pos
        G.ndata["f"] = self.embedding(data.atomic_numbers.long()).reshape(
            -1, self.atom_feature_size, 1
        )
        if use_pbc:
            G.edata["d"] = pbc_dist_vec
        else:
            G.edata["d"] = data.pos[dst] - data.pos[src]

        return G
