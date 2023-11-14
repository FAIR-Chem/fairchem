"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from math import pi as PI
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import get_pbc_distances, radius_graph_pbc, conditional_grad
from ocpmodels.datasets.embeddings import ATOMIC_RADII, CONTINUOUS_EMBEDDINGS
from ocpmodels.models.base_model import BaseModel
from ocpmodels.models.utils.activations import Act
from ocpmodels.models.utils.basis import Basis, SphericalSmearing
from ocpmodels.models.utils.pos_encodings import PositionalEncoding
from ocpmodels.modules.phys_embeddings import PhysEmbedding
from ocpmodels.models.force_decoder import ForceDecoder


class FNDecoder(nn.Module):
    def __init__(self, decoder_type, decoder_activation_str, output_dim):
        super(FNDecoder, self).__init__()
        self.decoder_type = decoder_type
        self.decoder_activation = Act(decoder_activation_str)
        self.output_dim = output_dim

        if self.decoder_type == "linear":
            self.decoder = nn.Sequential(nn.Linear(self.output_dim, 3))
        elif self.decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim),
                nn.BatchNorm1d(self.output_dim),
                self.decoder_activation,
                nn.Linear(self.output_dim, 3),
            )
        else:
            raise ValueError(f"Undefined force decoder: {self.decoder_type}")

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.decoder(x)


class InteractionBlock(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        mlp_basis_dim,
        basis_type,
        depth_mlp_edge=2,
        depth_mlp_trans=1,
        activation_str="ssp",
        ablation="none",
    ):
        super(InteractionBlock, self).__init__(aggr="add")

        self.activation = Act(activation_str)
        self.ablation = ablation
        self.basis_type = basis_type

        # basis function assumes input is in the range of [-1,1]
        if self.basis_type != "rawcat":
            self.lin_basis = torch.nn.Linear(mlp_basis_dim, hidden_channels)

        if self.ablation == "nocond":
            # the edge filter only depends on edge_attr
            in_features = (
                mlp_basis_dim if self.basis_type == "rawcat" else hidden_channels
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
                mlp_edge.append(self.activation)
                mlp_edge.append(torch.nn.Linear(hidden_channels, hidden_channels))
        else:
            ## need batch normalization afterwards. Otherwise training is unstable.
            mlp_edge = [
                torch.nn.Linear(in_features, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
            ]
        self.mlp_edge = torch.nn.Sequential(*mlp_edge)

        if not self.ablation == "nofilter":
            self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

        if depth_mlp_trans > 0:
            mlp_trans = [torch.nn.Linear(hidden_channels, hidden_channels)]
            for i in range(depth_mlp_trans):
                mlp_trans.append(torch.nn.BatchNorm1d(hidden_channels))
                mlp_trans.append(self.activation)
                mlp_trans.append(torch.nn.Linear(hidden_channels, hidden_channels))
        else:
            # need batch normalization afterwards. Otherwise, becomes NaN
            mlp_trans = [
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
            ]

        self.mlp_trans = torch.nn.Sequential(*mlp_trans)

        if not self.ablation == "noself":
            self.center_W = torch.nn.Parameter(torch.Tensor(1, hidden_channels))

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
            emb = torch.cat([edge_emb, x[edge_index[0]], x[edge_index[1]]], dim=1)

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


# flake8: noqa: C901
@registry.register_model("forcenet")
class ForceNet(BaseModel):
    r"""Implementation of ForceNet architecture.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        use_pbc (bool, optional): Use of periodic boundary conditions.
            (default: true)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        graph_rewiring (str, optional): Method used to create the graph,
            among "", remove-tag-0, supernodes.
        energy_head (str, optional): Method to compute energy prediction
            from atom representations.
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`512`)
        tag_hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`512`)
        pg_hidden_channels (int, optional): Hidden period and group embed size.
            (default: obj:`32`)
        phys_embed (bool, optional): Concat fixed physics-aware embeddings.
        phys_hidden_channels (int, optional): Hidden size of learnable phys embed.
            (default: obj:`32`)
        num_interactions (int, optional): Number of interaction blocks.
            (default: :obj:`5`)
        feat (str, optional): Input features to be used
            (default: :obj:`full`)
        num_freqs (int, optional): Number of frequencies for basis function.
            (default: :obj:`50`)
        max_n (int, optional): Maximum order of spherical harmonics.
            (default: :obj:`6`)
        basis (str, optional): Basis function to be used.
            (default: :obj:`full`)
        depth_mlp_edge (int, optional): Depth of MLP for edges in interaction blocks.
            (default: :obj:`2`)
        depth_mlp_node (int, optional): Depth of MLP for nodes in interaction blocks.
            (default: :obj:`1`)
        activation_str (str, optional): Activation function used post linear layer in all message passing MLPs.
            (default: :obj:`swish`)
        ablation (str, optional): Type of ablation to be performed.
            (default: :obj:`none`)
        decoder_hidden_channels (int, optional): Number of hidden channels in the decoder.
            (default: :obj:`512`)
        decoder_type (str, optional): Type of decoder: linear or MLP.
            (default: :obj:`mlp`)
        decoder_activation_str (str, optional): Activation function used post linear layer in decoder.
            (default: :obj:`swish`)
        training (bool, optional): If set to :obj:`True`, specify training phase.
            (default: :obj:`True`)
    """

    def __init__(self, **kwargs):
        super(ForceNet, self).__init__()
        self.ablation = kwargs["ablation"]
        self.basis = kwargs["basis"]
        self.cutoff = kwargs["cutoff"]
        self.energy_head = kwargs["energy_head"]
        self.feat = kwargs["feat"]
        self.otf_graph = kwargs["otf_graph"]
        self.regress_forces = kwargs["regress_forces"]
        self.use_pbc = kwargs["use_pbc"]

        self.use_tag = kwargs["tag_hidden_channels"] > 0
        self.use_pg = kwargs["pg_hidden_channels"] > 0
        self.use_positional_embeds = kwargs["graph_rewiring"] in {
            "one-supernode-per-graph",
            "one-supernode-per-atom-type",
            "one-supernode-per-atom-type-dist",
        }
        self.use_mlp_phys = kwargs["phys_hidden_channels"] > 0 and kwargs["phys_embeds"]

        if self.ablation not in [
            "none",
            "nofilter",
            "nocond",
            "nodistlist",
            "onlydist",
            "nodelinear",
            "edgelinear",
            "noself",
        ]:
            raise ValueError(f"Unknown ablation called {self.ablation}.")

        """
        Descriptions of ablations:
            - none: base ForceNet model
            - nofilter: no element-wise filter parameterization in message modeling
            - nocond: convolutional filter is only conditioned on edge features, not
                node embeddings
            - nodistlist: no atomic radius information in edge features
            - onlydist: edge features only contains distance information. Orientation
                information is ommited.
            - nodelinear: node update MLP function is replaced with linear function
                followed by batch normalization
            - edgelinear: edge MLP transformation function is replaced with linear
                function followed by batch normalization.
            - noself: no self edge of m_t.
        """

        assert (
            kwargs["tag_hidden_channels"] + 2 * kwargs["pg_hidden_channels"] + 16
            < kwargs["hidden_channels"]
        )

        if self.ablation == "edgelinear":
            kwargs["depth_mlp_edge"] = 0

        if self.ablation == "nodelinear":
            kwargs["depth_mlp_node"] = 0

        # read atom map and atom radii
        atom_map = torch.zeros(101, 9)
        for i in range(101):
            atom_map[i] = torch.tensor(CONTINUOUS_EMBEDDINGS[i])

        atom_radii = torch.zeros(101)
        for i in range(101):
            atom_radii[i] = ATOMIC_RADII[i]
        atom_radii = atom_radii / 100

        self.atom_radii = nn.Parameter(atom_radii, requires_grad=False)
        self.basis_type = self.basis

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
                max_n=kwargs["max_n"], option=self.pbc_sph_option
            )

        # Tag embedding
        if self.use_tag:
            self.tag_embedding = nn.Embedding(3, kwargs["tag_hidden_channels"])

        # Phys embeddings
        self.phys_emb = PhysEmbedding(props=kwargs["phys_embeds"], pg=self.use_pg)
        if self.use_mlp_phys:
            self.phys_lin = nn.Linear(
                self.phys_emb.n_properties, kwargs["phys_hidden_channels"]
            )
        else:
            kwargs["phys_hidden_channels"] = self.phys_emb.n_properties

        # Period + group embeddings
        if self.use_pg:
            self.period_embedding = nn.Embedding(
                self.phys_emb.period_size, kwargs["pg_hidden_channels"]
            )
            self.group_embedding = nn.Embedding(
                self.phys_emb.group_size, kwargs["pg_hidden_channels"]
            )

        # Position encoding
        if self.use_positional_embeds:
            self.pe = PositionalEncoding(kwargs["hidden_channels"], 210)

        # Main embedding
        if self.feat == "simple":
            self.embedding = nn.Embedding(
                100,
                kwargs["hidden_channels"]
                - kwargs["tag_hidden_channels"]
                - kwargs["phys_hidden_channels"]
                - 2 * kwargs["pg_hidden_channels"],
            )
            # self.embedding = nn.Embedding(100, kwargs["hidden_channels"])

            # set up dummy atom_map that only contains atomic_number information
            atom_map = torch.linspace(0, 1, 101).view(-1, 1).repeat(1, 9)
            self.atom_map = nn.Parameter(atom_map, requires_grad=False)

        elif self.feat == "full":
            # Normalize along each dimension
            atom_map[0] = np.nan
            atom_map_notnan = atom_map[atom_map[:, 0] == atom_map[:, 0]]
            atom_map_min = torch.min(atom_map_notnan, dim=0)[0]
            atom_map_max = torch.max(atom_map_notnan, dim=0)[0]
            atom_map_gap = atom_map_max - atom_map_min

            ## squash to [0,1]
            atom_map = (atom_map - atom_map_min.view(1, -1)) / atom_map_gap.view(1, -1)

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
                num_freqs=kwargs["num_freqs"],
                basis_type=node_basis_type,
                act=kwargs["activation_str"],
            )
            self.embedding = torch.nn.Sequential(
                basis,
                torch.nn.Linear(
                    basis.out_dim,
                    kwargs["hidden_channels"]
                    - kwargs["tag_hidden_channels"]
                    - kwargs["phys_hidden_channels"]
                    - 2 * kwargs["pg_hidden_channels"],
                ),
            )

        else:
            raise ValueError("Undefined feature type for atom")

        # process basis function for edge feature
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
                logging.info(
                    "Under onlydist ablation, spherical basis is reduced to powersine basis."
                )
                self.basis_type = "powersine"
                self.pbc_sph = None

        else:
            in_feature = 7
        self.basis_fun = Basis(
            in_feature,
            kwargs["num_freqs"],
            self.basis_type,
            kwargs["activation_str"],
            sph=self.pbc_sph,
        )

        # process interaction blocks
        self.interactions = torch.nn.ModuleList()
        for _ in range(kwargs["num_interactions"]):
            block = InteractionBlock(
                kwargs["hidden_channels"],
                self.basis_fun.out_dim,
                self.basis_type,
                depth_mlp_edge=kwargs["depth_mlp_edge"],
                depth_mlp_trans=kwargs["depth_mlp_node"],
                activation_str=kwargs["activation_str"],
                ablation=self.ablation,
            )
            self.interactions.append(block)

        self.lin = torch.nn.Linear(
            kwargs["hidden_channels"], kwargs["decoder_hidden_channels"]
        )
        self.activation = Act(kwargs["activation_str"])

        # Force head
        self.decoder = (
            ForceDecoder(
                kwargs["force_decoder_type"],
                kwargs["decoder_hidden_channels"],
                kwargs["force_decoder_model_config"],
                self.activation,
            )
            if "direct" in self.regress_forces
            else None
        )

        #  weighted average blocks
        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            self.w_lin = nn.Linear(kwargs["hidden_channels"], 1)

        # Projection layer for energy prediction
        self.energy_mlp = nn.Linear(kwargs["decoder_hidden_channels"], 1)

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        # Rewire the graph
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

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
                return_distance_vec=True,
            )
            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            edge_vec = out["distance_vec"]
        else:
            edge_index = data.edge_index
            edge_dist = data.distances
            edge_vec = pos[edge_index[0, :]] - pos[edge_index[1, :]]

        if self.feat == "simple":
            h = self.embedding(z)

        elif self.feat == "full":
            h = self.embedding(self.atom_map[z])
        else:
            raise RuntimeError("Undefined feature type for atom")

        if self.phys_emb.device != batch.device:
            self.phys_emb = self.phys_emb.to(batch.device)

        if self.use_tag:
            assert data.tags is not None
            h_tag = self.tag_embedding(data.tags)
            h = torch.cat((h, h_tag), dim=1)

        if self.phys_emb.n_properties > 0:
            h_phys = self.phys_emb.properties[z]
            if self.use_mlp_phys:
                h_phys = self.phys_lin(h_phys)
            h = torch.cat((h, h_phys), dim=1)

        if self.use_pg:
            assert self.phys_emb.period is not None
            h_period = self.period_embedding(self.phys_emb.period[z])
            h_group = self.group_embedding(self.phys_emb.group[z])
            h = torch.cat((h, h_period, h_group), dim=1)

        if self.use_positional_embeds:
            idx_of_non_zero_val = (data.tags == 0).nonzero().T.squeeze(0)
            h_pos = torch.zeros_like(h, device=h.device)
            h_pos[idx_of_non_zero_val, :] = self.pe(data.subnodes).to(
                device=h_pos.device
            )
            h += h_pos

        if self.pbc_apply_sph_harm:
            edge_vec_normalized = edge_vec / edge_dist.view(-1, 1)
            edge_attr_sph = self.pbc_sph(edge_vec_normalized)

        # calculate the edge weight according to the dist
        edge_weight = torch.cos(0.5 * edge_dist * PI / self.cutoff)

        # normalized edge vectors
        edge_vec_normalized = edge_vec / edge_dist.view(-1, 1)

        # edge distance, taking the atom_radii into account
        # each element lies in [0,1]
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

        # make sure distance is positive
        edge_dist_list[edge_dist_list < 1e-3] = 1e-3

        # squash to [0,1] for gaussian basis
        if self.basis_type == "gauss":
            edge_vec_normalized = (edge_vec_normalized + 1) / 2.0

        # process raw_edge_attributes to generate edge_attributes
        if self.ablation == "onlydist":
            raw_edge_attr = edge_dist_list
        else:
            raw_edge_attr = torch.cat([edge_vec_normalized, edge_dist_list], dim=1)

        if "sph" in self.basis_type:
            edge_attr = self.basis_fun(raw_edge_attr, edge_attr_sph)
        else:
            edge_attr = self.basis_fun(raw_edge_attr)

        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)

        # pass edge_attributes through interaction blocks
        for i, interaction in enumerate(self.interactions):
            h = h + interaction(h, edge_index, edge_attr, edge_weight)

        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        # MLPs
        h = self.lin(h)
        h = self.activation(h)

        # Weighted average
        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        out = scatter(h, batch, dim=0, reduce="add")

        energy = self.energy_mlp(out)

        preds = {"energy": energy, "hidden_state": h}

        return preds

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        assert "direct" in self.regress_forces
        return self.decoder(preds["hidden_state"])
