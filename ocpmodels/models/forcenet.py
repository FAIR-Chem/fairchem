import os
from math import pi as PI

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool, radius_graph
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import get_pbc_distances, radius_graph_pbc

from .activations import Act
from .basis import Basis, SphericalSmearing


# flake8: noqa: C901
@registry.register_model("forcenet")
class ForceNet(nn.Module):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        hidden_channels=128,
        num_interactions=3,
        cutoff=6.0,
        feat="full",
        num_freqs=50,
        max_n=10,
        basis="powersine",
        depth_mlp_edge=2,
        depth_mlp_node=1,
        act="ssp",
        ablation="None",
        sph=None,
        decoder_final_channels=64,
        decoder_type="mlp",
        training=True,
        decoder_act="relu",
        otf_graph=False,
    ):

        print(hidden_channels, num_interactions, cutoff, feat, num_freqs)
        super(ForceNet, self).__init__()

        print(hidden_channels, num_interactions, cutoff, feat, num_freqs)

        if ablation not in [
            "None",
            "nofilter",
            "nocond",
            "nodistlist",
            "onlydist",
            "nodelinear",
            "edgelinear",
            "noself",
        ]:
            raise ValueError(f"Unknown ablation called {ablation}.")

        """
        Descriptions of ablations:
            - None: base ForceNet model
            - nofilter: no element-wise filter parameterization in message modeling
            - nocond: convolutional filter is only conditioned on edge features, not node embeddings
            - nodistlist: no atomic radius information in edge features
            - onlydist: edge features only contains distance information. Orientation information is ommited.
            - nodelinear: node update MLP function is replaced with linear function followed by batch normalization
            - edgelinear: edge MLP transformation function is replaced with linear function followed by batch normalization.
            - noself: no self edge of m_t.
        """

        if ablation != "None":
            print(f"Ablation study on {ablation}.")

        self.training = training
        self.ablation = ablation
        self.otf_graph = otf_graph

        self.cutoff = cutoff
        self.output_dim = decoder_final_channels
        print("self.output_dim =", self.output_dim, hidden_channels)
        self.feat = feat
        self.num_freqs = num_freqs
        self.num_layers = num_interactions
        self.max_n = max_n
        self.sph = sph

        if self.ablation == "edgelinear":
            depth_mlp_edge = 0

        if self.ablation == "nodelinear":
            depth_mlp_node = 0

        ### reading atom radii
        path = os.path.join(os.path.dirname(__file__), "embeddings")
        atom_map = torch.stack(
            torch.load(os.path.join(path, "cgcnn_embeddings.pt"))
        )
        self.missing_ind = torch.load(os.path.join(path, "missing_radii.pt"))

        atom_radii = (
            torch.stack(
                [torch.tensor([np.nan])]
                + torch.load(os.path.join(path, "atomic_radii.pt"))[1:]
            ).view(
                -1,
            )
            / 100
        )
        atom_radii[self.missing_ind] = np.nan
        self.atom_radii = nn.Parameter(atom_radii, requires_grad=False)
        self.basis_type = basis

        self.pbc_apply_sph_harm = "sph" in self.basis_type
        self.pbc_sph_option = None
        # for spherical harmonics for PBC
        if "sphall" in self.basis_type:
            self.pbc_sph_option = "all"
        elif "sphsine" in self.basis_type:
            self.pbc_sph_option = "sine"
        elif "sphcosine" in self.basis_type:
            self.pbc_sph_option = "cosine"

        self.pbc_sph = None
        if self.pbc_apply_sph_harm:
            self.pbc_sph = SphericalSmearing(
                max_n=self.max_n, option=self.pbc_sph_option
            )

        if self.feat == "simple":
            self.embedding = nn.Embedding(100, hidden_channels)

            # set up dummy atom_map that only contains atomic_number information
            atom_map = torch.linspace(0, 1, 101).view(-1, 1).repeat(1, 9)
            atom_map[self.missing_ind] = np.nan
            self.atom_map = nn.Parameter(atom_map, requires_grad=False)

        elif self.feat == "full":
            # Normalize along each dimaension
            atom_map[self.missing_ind] = np.nan
            atom_map[0] = np.nan
            atom_map_notnan = atom_map[atom_map[:, 0] == atom_map[:, 0]]
            atom_map_min = torch.min(atom_map_notnan, dim=0)[0]
            atom_map_max = torch.max(atom_map_notnan, dim=0)[0]
            atom_map_gap = atom_map_max - atom_map_min

            ## squash to [0,1]
            atom_map = (
                atom_map - atom_map_min.view(1, -1)
            ) / atom_map_gap.view(1, -1)

            self.atom_map = torch.nn.Parameter(atom_map, requires_grad=False)

            in_features = 9
            # first apply basis function and then linear function
            if "sph" in self.basis_type:
                # spherical basis is only meaningful for edge feature, so use powersine instead
                node_basis_type = "powersine"
            else:
                node_basis_type = self.basis_type
            basis = Basis(
                in_features,
                num_freqs=num_freqs,
                basis_type=node_basis_type,
                act=act,
            )
            self.embedding = torch.nn.Sequential(
                basis, torch.nn.Linear(basis.out_dim, hidden_channels)
            )

        else:
            raise ValueError("Undefined feature type for atom")

        if self.ablation == "nodistlist":
            # do not consider additional distance edge features
            # normalized (x,y,z) + distance
            in_feature = 4
        elif self.ablation == "onlydist":
            # only consider distance-based edge features
            # ignore normalized (x,y,z)
            in_feature = 4

            # if basis_type is spherical harmonics, then reduce to powersine
            if "sph" in self.basis_type:
                print(
                    "Under onlydist ablation, spherical basis is reduced to powersine basis."
                )
                self.basis_type = "powersine"
                sph = None

        else:
            in_feature = 7
        # basis function for edge feature
        self.basis_fun = Basis(
            in_feature, num_freqs, self.basis_type, act, sph=sph
        )

        self.interactions = torch.nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels,
                self.basis_fun.out_dim,
                self.basis_type,
                depth_mlp_edge=depth_mlp_edge,
                depth_mlp_trans=depth_mlp_node,
                act=act,
                ablation=ablation,
            )
            self.interactions.append(block)

        self.lin = torch.nn.Linear(hidden_channels, self.output_dim)
        self.act = Act(act)

        # decoder part of ForceNet
        self.decoder_type = decoder_type
        self.decoder_act = Act(decoder_act)

        if self.decoder_type == "linear":
            self.decoder = nn.Sequential(nn.Linear(self.output_dim, 3))
        elif self.decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim),
                nn.BatchNorm1d(self.output_dim),
                self.decoder_act,
                nn.Linear(self.output_dim, 3),
            )
        else:
            raise ValueError(f"Undefined force decoder: {self.decoder_type}")

        # Projection layer for energy prediction
        self.energy_mlp = nn.Linear(self.output_dim, 1)

        # initializating decoder weights TODO(sid): maybe move to reset_parameters
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, data):
        z = data.atomic_numbers.long()

        pos = data.pos
        batch = data.batch

        print(pos.shape)
        if self.feat == "simple":
            h = self.embedding(z)
        elif self.feat == "full":
            h = self.embedding(self.atom_map[z])
        else:
            raise RuntimeError("Undefined feature type for atom")

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50, data.pos.device
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        # assert self.use_pbc
        out = get_pbc_distances(
            pos,
            data.edge_index,
            data.cell,
            data.cell_offsets,
            data.neighbors,
            return_distance_vec=True,
        )

        edge_index = out["edge_index"]
        edge_dist = out["distances"]
        edge_vec = out["distance_vec"]

        if self.pbc_apply_sph_harm:
            edge_vec_normalized = edge_vec / edge_dist.view(-1, 1)
            edge_attr_sph = self.pbc_sph(edge_vec_normalized)

        ### calculate the edge weight according to the dist
        edge_weight = torch.cos(0.5 * edge_dist * PI / self.cutoff)

        edge_vec_normalized = edge_vec / edge_dist.view(-1, 1)

        ## edge distance, taking the atom_radii into account
        ## each element lies in [0,1]
        edge_dist_list = (
            torch.stack(
                [
                    edge_dist,
                    edge_dist - self.atom_radii[z[edge_index[0]]],
                    edge_dist - self.atom_radii[z[edge_index[1]]],
                    edge_dist
                    - self.atom_radii[z[edge_index[0]]]
                    - self.atom_radii[z[edge_index[1]]],
                ]
            ).transpose(0, 1)
            / self.cutoff
        )

        if self.ablation == "nodistlist":
            edge_dist_list = edge_dist_list[:, 0].view(-1, 1)

        ## make sure distance is positive
        edge_dist_list[edge_dist_list < 1e-3] = 1e-3

        # squash to [0,1] for gaussian basis
        if self.basis_type == "gauss":
            edge_vec_normalized = (edge_vec_normalized + 1) / 2.0

        if self.ablation == "onlydist":
            raw_edge_attr = edge_dist_list
        else:
            raw_edge_attr = torch.cat(
                [edge_vec_normalized, edge_dist_list], dim=1
            )

        if "sph" in self.basis_type:
            edge_attr = self.basis_fun(raw_edge_attr, edge_attr_sph)
        else:
            edge_attr = self.basis_fun(raw_edge_attr)

        for i, interaction in enumerate(self.interactions):
            h = h + interaction(h, edge_index, edge_attr, edge_weight)

        h = self.lin(h)
        h = self.act(h)

        out = scatter(h, batch, dim=0, reduce="add")

        force = self.decoder(h)
        energy = self.energy_mlp(out)
        return energy, force

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class InteractionBlock(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        mlp_basis_dim,
        basis_type,
        depth_mlp_edge=2,
        depth_mlp_trans=1,
        act="ssp",
        ablation="None",
    ):
        super(InteractionBlock, self).__init__(aggr="add")

        self.act = Act(act)
        self.ablation = ablation
        self.basis_type = basis_type

        # basis function assumes input is in the range of [-1,1]
        if self.basis_type != "rawcat":
            self.lin_basis = torch.nn.Linear(mlp_basis_dim, hidden_channels)

        if self.ablation == "nocond":
            ## the edge filter only depends on edge_attr
            in_features = (
                mlp_basis_dim
                if self.basis_type == "rawcat"
                else hidden_channels
            )
        else:
            # edge filter depends on edge_attr and current node embedding
            in_features = (
                mlp_basis_dim + 2 * hidden_channels
                if self.basis_type == "rawcat"
                else 3 * hidden_channels
            )

        if depth_mlp_edge > 0:
            mlp_edge = [torch.nn.Linear(in_features, hidden_channels)]
            for i in range(depth_mlp_edge):
                mlp_edge.append(self.act)
                mlp_edge.append(
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )
        else:
            print("edgelinear")
            ## need batch normalization afterwards. Otherwise training is unstable.
            mlp_edge = [
                torch.nn.Linear(in_features, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
            ]
        self.mlp_edge = torch.nn.Sequential(*mlp_edge)

        print(f"depth of mlp_edge is {depth_mlp_edge}")

        if not self.ablation == "nofilter":
            self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

        if depth_mlp_trans > 0:
            mlp_trans = [torch.nn.Linear(hidden_channels, hidden_channels)]
            for i in range(depth_mlp_trans):
                mlp_trans.append(torch.nn.BatchNorm1d(hidden_channels))
                mlp_trans.append(self.act)
                mlp_trans.append(
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )
        else:
            print("nodelinear")
            ## need batch normalization afterwards. Otherwise, becomes NaN
            mlp_trans = [
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
            ]

        self.mlp_trans = torch.nn.Sequential(*mlp_trans)
        print(f"depth of mlp_trans is {depth_mlp_trans}")

        if not self.ablation == "noself":
            self.center_W = torch.nn.Parameter(
                torch.Tensor(1, hidden_channels)
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.basis_type != "rawcat":
            torch.nn.init.xavier_uniform_(self.lin_basis.weight)
            self.lin_basis.bias.data.fill_(0)

        for m in self.mlp_trans:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        for m in self.mlp_edge:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        if not self.ablation == "nofilter":
            torch.nn.init.xavier_uniform_(self.lin.weight)
            self.lin.bias.data.fill_(0)

        if not self.ablation == "noself":
            torch.nn.init.xavier_uniform_(self.center_W)

    def forward(self, x, edge_index, edge_attr, edge_weight):
        if self.basis_type != "rawcat":
            edge_emb = self.lin_basis(edge_attr)
        else:
            # for rawcat, we directly use the raw feature
            edge_emb = edge_attr

        if self.ablation == "nocond":
            emb = edge_emb
        else:
            emb = torch.cat(
                [edge_emb, x[edge_index[0]], x[edge_index[1]]], dim=1
            )

        W = self.mlp_edge(emb) * edge_weight.view(-1, 1)
        if self.ablation == "nofilter":
            x = self.propagate(edge_index, x=x, W=W) + self.center_W
        else:
            x = self.lin(x)
            if self.ablation == "noself":
                x = self.propagate(edge_index, x=x, W=W)
            else:
                x = self.propagate(edge_index, x=x, W=W) + self.center_W * x
        x = self.mlp_trans(x)

        return x

    def message(self, x_j, W):
        if self.ablation == "nofilter":
            return W
        else:
            return x_j * W
