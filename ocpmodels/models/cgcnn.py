import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing

from ocpmodels.common.registry import registry
from ocpmodels.datasets.elemental_embeddings import EMBEDDINGS
from ocpmodels.models.base import BaseModel
from ocpmodels.modules.layers import CGCNNConv


@registry.register_model("cgcnn")
class CGCNN(BaseModel):
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
        cutoff=6.0,
        num_gaussians=50,
    ):
        super(CGCNN, self).__init__(num_atoms, bond_feat_dim, num_targets)
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc

        # Get CGCNN atom embeddings
        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(EMBEDDINGS[i + 1])
        self.embedding_fc = nn.Linear(92, atom_embedding_size)

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

    def forward(self, data):
        # Get node features
        if self.embedding.device != data.atomic_numbers.device:
            self.embedding = self.embedding.to(data.atomic_numbers.device)
        data.x = self.embedding[data.atomic_numbers.long() - 1]

        if self.use_pbc:
            pos = data.pos
            edge_index = data.edge_index
            row, col = edge_index

            edge_weight = pos[row] - pos[col]

            # correct for pbc
            cell = torch.repeat_interleave(data.cell, data.natoms * 12, dim=0)
            cell_offsets = data.cell_offsets
            offsets = (
                cell_offsets.float()
                .view(-1, 1, 3)
                .bmm(cell.float())
                .view(-1, 3)
            )
            edge_weight += offsets

            # compute distances
            edge_weight = edge_weight.norm(dim=-1)

            # remove zero distances
            nonzero_idx = torch.nonzero(edge_weight).flatten()
            edge_index = edge_index[:, nonzero_idx]
            # remove -1 indices
            nonnegative_idx = (edge_index[1] != -1).nonzero().view(-1)
            edge_index = edge_index[:, nonnegative_idx]
            data.edge_index = edge_index
        else:
            data.edge_index = radius_graph(
                data.pos, r=self.cutoff, batch=data.batch
            )
        row, col = data.edge_index
        pos = data.pos
        if self.regress_forces:
            pos = pos.requires_grad_(True)
        data.edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        data.edge_attr = self.distance_expansion(data.edge_weight)

        # Forward pass through the network
        mol_feats = self._convolve(data)
        mol_feats = self.conv_to_fc(mol_feats)
        if hasattr(self, "fcs"):
            mol_feats = self.fcs(mol_feats)

        energy = self.fc_out(mol_feats)
        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    pos,
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
            node_feats = f(
                node_feats, data.edge_index, data.edge_weight, data.edge_attr
            )
        mol_feats = global_mean_pool(node_feats, data.batch)
        return mol_feats
