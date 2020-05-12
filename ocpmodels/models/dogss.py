import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from ocpmodels.common.registry import registry
from ocpmodels.models.base import BaseModel
from ocpmodels.modules.layers import DOGSSConv


@registry.register_model("dogss")
class DOGSS(BaseModel):
    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        atom_embedding_size=64,
        num_graph_conv_layers=6,
        num_dist_layers=0,
        num_const_layers=0,
        fc_feat_size=128,
        dist_feat_dim=128,
        const_feat_dim=128,
        D_feat_dim=128,
        max_num_nbr=12,
        energy_mode="Harmonic",
        max_opt_steps=300,
        min_opt_steps=10,
        opt_step_size=0.3,
        momentum=0.8,
    ):
        super(DOGSS, self).__init__(num_atoms, bond_feat_dim, num_targets)

        self.max_num_nbr = max_num_nbr
        self.max_opt_steps = max_opt_steps
        self.min_opt_steps = min_opt_steps
        self.opt_step_size = opt_step_size
        self.momentum = momentum
        self.energy_mode = energy_mode

        self.embedding = nn.Linear(self.num_atoms, atom_embedding_size)

        self.convs = nn.ModuleList(
            [
                DOGSSConv(
                    node_dim=atom_embedding_size, edge_dim=self.bond_feat_dim
                )
                for _ in range(num_graph_conv_layers)
            ]
        )

        self.conv_to_bond_distance = nn.Linear(
            2 * atom_embedding_size + bond_feat_dim, dist_feat_dim
        )
        self.bond_distance_bn = nn.BatchNorm1d(dist_feat_dim)

        self.conv_to_bond_constant = nn.Linear(
            2 * atom_embedding_size + bond_feat_dim, const_feat_dim
        )
        self.bond_constant_bn = nn.BatchNorm1d(const_feat_dim)

        self.softplus = nn.Softplus()

        if num_dist_layers > 1:
            layers_dist = []
            for i in range(num_dist_layers - 1):
                layers_dist.append(nn.Linear(dist_feat_dim, dist_feat_dim))
                layers_dist.append(nn.BatchNorm1d(dist_feat_dim))
                layers_dist.append(nn.Softplus())
            self.layers_dist = nn.Sequential(*layers_dist)
            self.bond_distance = nn.Linear(dist_feat_dim, 1)

        if num_const_layers > 1:
            layers_const = []
            for i in range(num_const_layers - 1):
                layers_const.append(nn.Linear(const_feat_dim, const_feat_dim))
                layers_const.append(nn.BatchNorm1d(const_feat_dim))
                layers_const.append(nn.Softplus())
            self.layers_const = nn.Sequential(*layers_const)
            self.bond_constant = nn.Linear(const_feat_dim, 1)

    def forward(self, data):
        atom_pos = data.atom_pos.requires_grad_(True)
        cells = data.cells
        nbr_fea_offset = data.nbr_fea_offset
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        free_atom_idx = np.where(data.fixed_base.cpu().numpy() == 0)[0]
        fixed_atom_idx = np.where(data.fixed_base.cpu().numpy() == 1)[0]
        # ads_idx = np.where(data.ads_tag.cpu().numpy() == 1)[0]

        atom_fea = self.embedding(data.x)
        for conv in self.convs:
            atom_fea, edge_attr = conv(
                x=atom_fea, edge_index=edge_index, edge_attr=edge_attr
            )

        distance = self.get_distance(
            atom_pos, cells, edge_index, nbr_fea_offset
        )  # N x 1

        bond_fea = torch.cat(
            (atom_fea[edge_index[0]], atom_fea[edge_index[1]], edge_attr),
            dim=1,
        )

        # Network to learn equilibrium spring bond distances
        bond_dist_fea = bond_fea
        bond_distance = self.conv_to_bond_distance(bond_dist_fea)
        if hasattr(self, "layers_dist"):
            bond_distance = self.softplus(self.bond_distance_bn(bond_distance))
            bond_distance = self.layers_dist(bond_distance)
            bond_distance = self.softplus(
                (self.bond_distance(bond_distance) + distance)
            )
        else:
            bond_distance = self.softplus(
                self.bond_distance_bn(bond_distance) + distance
            )
            bond_distance = torch.mean(bond_distance, dim=1).unsqueeze(-1)

        # Netwok to learn spring constants
        bond_const_fea = bond_fea
        bond_constant = self.conv_to_bond_constant(bond_const_fea)
        if hasattr(self, "layers_const"):
            bond_constant = self.layers_const(bond_constant)
            bond_constant = self.softplus(
                self.bond_constant(bond_constant)
            ) / len(bond_constant)
        else:
            bond_constant = self.softplus(self.bond_constant_bn(bond_constant))
            bond_constant = torch.mean(bond_constant, dim=1).unsqueeze(
                -1
            ) / len(bond_constant)

        V = torch.tensor(0.0)
        beta = self.momentum
        steepest_descent_step = torch.FloatTensor([1.0])
        grad = torch.FloatTensor([100.0])
        step_count = 0

        while (
            torch.max(torch.abs(steepest_descent_step)) > 0.0005
            and step_count < self.max_opt_steps
        ) or step_count < self.min_opt_steps:

            distance = self.get_distance(
                atom_pos, cells, edge_index, nbr_fea_offset
            )
   
            if self.energy_mode == "Harmonic":
                potential_E = bond_constant * (bond_distance - distance) ** 2


            grad_E = potential_E.sum()

            grad = torch.autograd.grad(
                grad_E, atom_pos, retain_graph=True, create_graph=True
            )[0]
            grad[fixed_atom_idx] = 0

            if torch.isinf(grad_E).item():
                print("grad_E becomes inf")

            # detect if step is going off the rails
            if torch.max(torch.isnan(grad)) == 1:
                print("nan")
                return atom_pos[free_atom_idx]
            if torch.max(torch.abs(self.opt_step_size * grad)) > 5.0:
                print("blow up")
                return atom_pos[free_atom_idx]

            steepest_descent_step = -self.opt_step_size * grad

            # Momentum
            V = beta * V + (1 - beta) * grad
            atom_pos = atom_pos - self.opt_step_size * V
            step_count += 1

            del grad, potential_E

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return atom_pos[free_atom_idx]

    def get_distance(self, atom_pos, cells, edge_index, nbr_fea_offset):
        differ = (
            atom_pos[edge_index[1]]
            - atom_pos[edge_index[0]]
            + torch.bmm(
                nbr_fea_offset.view(-1, self.max_num_nbr, 3), cells
            ).view(-1, 3)
        )
        differ_sum = torch.sum(differ ** 2, dim=1)
        differ_sum = torch.clamp(differ_sum, min=1e-6)
        distance = torch.sqrt(differ_sum).unsqueeze(-1)
        return distance  # N, 1
