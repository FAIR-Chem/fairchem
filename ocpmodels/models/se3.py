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


@registry.register_model("se3")
class SE3Transformer(nn.Module):
    """
    SE(3) equivariant GCN with attention from the
    `"SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks"
    <https://arxiv.org/pdf/2006.10503.pdf>`

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
        num_degrees: int = 4,
        edge_dim: int = 0,
        div: float = 4,
        pooling: str = "avg",
        n_heads: int = 1,
        use_pbc: bool = True,
        regress_forces: bool = True,
    ):
        super().__init__()
        # Build the network
        self.atom_feature_size = atom_feature_size
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces

        self.fibers = {
            "in": Fiber(1, atom_feature_size),
            "mid": Fiber(num_degrees, self.num_channels),
            "out": Fiber(1, num_degrees * self.num_channels),
        }

        self.embedding = Embedding(100, atom_feature_size)

        blocks = self._build_gcn(self.fibers, 1)

        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers["in"]
        for i in range(self.num_layers):
            Gblock.append(
                GSE3Res(
                    fin,
                    fibers["mid"],
                    edge_dim=self.edge_dim,
                    div=self.div,
                    n_heads=self.n_heads,
                )
            )
            Gblock.append(GNormSE3(fibers["mid"]))
            fin = fibers["mid"]
        Gblock.append(
            GConvSE3(
                fibers["mid"],
                fibers["out"],
                self_interaction=True,
                edge_dim=self.edge_dim,
            )
        )

        # Pooling
        if self.pooling == "avg":
            Gblock.append(GAvgPooling())
        elif self.pooling == "max":
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(
            nn.Linear(
                self.fibers["out"].n_features, self.fibers["out"].n_features
            )
        )
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers["out"].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

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

        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
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
