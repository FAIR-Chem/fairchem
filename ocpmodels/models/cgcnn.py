"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.datasets.embeddings import KHOT_EMBEDDINGS, QMOF_KHOT_EMBEDDINGS
from ocpmodels.models.base import BaseModel


@registry.register_model("cgcnn")
class CGCNN(BaseModel):
    r"""Implementation of the Crystal Graph CNN model from the
    `"Crystal Graph Convolutional Neural Networks for an Accurate
    and Interpretable Prediction of Material Properties"
    <https://arxiv.org/abs/1710.10324>`_ paper.

    Args:
        num_atoms (int): Number of atoms.
        bond_feat_dim (int): Dimension of bond features.
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        atom_embedding_size (int, optional): Size of atom embeddings.
            (default: :obj:`64`)
        num_graph_conv_layers (int, optional): Number of graph convolutional layers.
            (default: :obj:`6`)
        fc_feat_size (int, optional): Size of fully connected layers.
            (default: :obj:`128`)
        num_fc_layers (int, optional): Number of fully connected layers.
            (default: :obj:`4`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        num_gaussians (int, optional): Number of Gaussians used for smearing.
            (default: :obj:`50.0`)
    """

    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        use_pbc=True,
        regress_forces=True,
        atom_embedding_size=64,
        num_graph_conv_layers=6,
        fc_feat_size=128,
        num_fc_layers=4,
        otf_graph=False,
        cutoff=6.0,
        num_gaussians=50,
        embeddings="khot",
    ):
        super(CGCNN, self).__init__(num_atoms, bond_feat_dim, num_targets)
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph

        # Get CGCNN atom embeddings
        if embeddings == "khot":
            embeddings = KHOT_EMBEDDINGS
        elif embeddings == "qmof":
            embeddings = QMOF_KHOT_EMBEDDINGS
        else:
            raise ValueError(
                'embedding mnust be either "khot" for original CGCNN K-hot elemental embeddings or "qmof" for QMOF K-hot elemental embeddings'
            )
        self.embedding = torch.zeros(100, len(embeddings[1]))
        for i in range(100):
            self.embedding[i] = torch.tensor(embeddings[i + 1])
        self.embedding_fc = nn.Linear(len(embeddings[1]), atom_embedding_size)

        self.convs = nn.ModuleList(
            [
                CGCNNConv(
                    node_dim=atom_embedding_size,
                    edge_dim=bond_feat_dim,
                    cutoff=cutoff,
                )
                for _ in range(num_graph_conv_layers)
            ]
        )

        self.conv_to_fc = nn.Sequential(
            nn.Linear(atom_embedding_size, fc_feat_size), nn.Softplus()
        )

        if num_fc_layers > 1:
            layers = []
            for _ in range(num_fc_layers - 1):
                layers.append(nn.Linear(fc_feat_size, fc_feat_size))
                layers.append(nn.Softplus())
            self.fcs = nn.Sequential(*layers)
        self.fc_out = nn.Linear(fc_feat_size, self.num_targets)

        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        # Get node features
        if self.embedding.device != data.atomic_numbers.device:
            self.embedding = self.embedding.to(data.atomic_numbers.device)
        data.x = self.embedding[data.atomic_numbers.long() - 1]

        pos = data.pos

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )

            data.edge_index = out["edge_index"]
            distances = out["distances"]
        else:
            data.edge_index = radius_graph(
                data.pos, r=self.cutoff, batch=data.batch
            )
            row, col = data.edge_index
            distances = (pos[row] - pos[col]).norm(dim=-1)

        data.edge_attr = self.distance_expansion(distances)
        # Forward pass through the network
        mol_feats = self._convolve(data)
        mol_feats = self.conv_to_fc(mol_feats)
        if hasattr(self, "fcs"):
            mol_feats = self.fcs(mol_feats)

        energy = self.fc_out(mol_feats)
        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    def _convolve(self, data):
        """
        Returns the output of the convolution layers before they are passed
        into the dense layers.
        """
        node_feats = self.embedding_fc(data.x)
        for f in self.convs:
            node_feats = f(node_feats, data.edge_index, data.edge_attr)
        mol_feats = global_mean_pool(node_feats, data.batch)
        return mol_feats


class CGCNNConv(MessagePassing):
    """Implements the message passing layer from
    `"Crystal Graph Convolutional Neural Networks for an
    Accurate and Interpretable Prediction of Material Properties"
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    """

    def __init__(self, node_dim, edge_dim, cutoff=6.0, **kwargs):
        super(CGCNNConv, self).__init__(aggr="add")
        self.node_feat_size = node_dim
        self.edge_feat_size = edge_dim
        self.cutoff = cutoff

        self.lin1 = nn.Linear(
            2 * self.node_feat_size + self.edge_feat_size,
            2 * self.node_feat_size,
        )
        self.bn1 = nn.BatchNorm1d(2 * self.node_feat_size)
        self.ln1 = nn.LayerNorm(self.node_feat_size)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)

        self.lin1.bias.data.fill_(0)

        self.bn1.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Arguments:
            x has shape [num_nodes, node_feat_size]
            edge_index has shape [2, num_edges]
            edge_attr is [num_edges, edge_feat_size]
        """
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, size=(x.size(0), x.size(0))
        )
        out = nn.Softplus()(self.ln1(out) + x)
        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Arguments:
            x_i has shape [num_edges, node_feat_size]
            x_j has shape [num_edges, node_feat_size]
            edge_attr has shape [num_edges, edge_feat_size]

        Returns:
            tensor of shape [num_edges, node_feat_size]
        """
        z = self.lin1(torch.cat([x_i, x_j, edge_attr], dim=1))
        z = self.bn1(z)
        z1, z2 = z.chunk(2, dim=1)
        z1 = nn.Sigmoid()(z1)
        z2 = nn.Softplus()(z2)
        return z1 * z2
