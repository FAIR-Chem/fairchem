"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from math import pi as PI
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.utils.pos_encodings import PositionalEncoding
from ocpmodels.modules.phys_embeddings import PhysEmbedding
from ocpmodels.modules.pooling import Graclus, Hierarchical_Pooling
from ocpmodels.preprocessing import (
    one_supernode_per_atom_type,
    one_supernode_per_atom_type_dist,
    one_supernode_per_graph,
    remove_tag0_nodes,
)

NUM_CLUSTERS = 20
NUM_POOLING_LAYERS = 1


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, self.mlp, cutoff
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class NewSchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        tag_hidden_channels (int, optional): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int, optional): Hidden period and group embed size.
            (default: obj:`32`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = "http://www.quantum-machine.org/datasets/trained_schnet_models.zip"

    def __init__(
        self,
        hidden_channels: int = 128,
        tag_hidden_channels: int = 32,
        pg_hidden_channels: int = 32,
        phys_embeds: bool = False,
        phys_hidden_channels: int = 32,
        graph_rewiring=False,
        energy_head=False,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        readout: str = "add",
        atomref: Optional[torch.Tensor] = None,
    ):
        super(NewSchNet, self).__init__()

        import ase

        self.hidden_channels = hidden_channels
        self.tag_hidden_channels = tag_hidden_channels
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_hidden_channels = phys_hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.scale = None
        self.use_tag = tag_hidden_channels > 0
        self.use_pg = pg_hidden_channels > 0
        self.use_mlp_phys = phys_hidden_channels > 0 and phys_embeds
        self.use_phys_embeddings = phys_embeds
        self.use_positional_embeds = graph_rewiring in {
            "one-supernode-per-graph",
            "one-supernode-per-atom-type",
            "one-supernode-per-atom-type-dist",
        }
        # self.use_positional_embeds = False
        self.energy_head = energy_head

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        # self.covalent_radii = torch.from_numpy(ase.data.covalent_radii)
        # self.vdw_radii = torch.from_numpy(ase.data.vdw_radii)
        self.register_buffer("atomic_mass", atomic_mass)

        if self.use_tag:
            self.tag_embedding = Embedding(3, tag_hidden_channels)

        # Phys embeddings
        self.phys_emb = PhysEmbedding(props=phys_embeds, pg=self.use_pg)
        if self.use_mlp_phys:
            self.phys_lin = Linear(
                self.phys_emb.n_properties, self.phys_hidden_channels
            )
        else:
            self.phys_hidden_channels = self.phys_emb.n_properties

        # Period + group embeddings
        if self.use_pg:
            self.period_embedding = Embedding(
                self.phys_emb.period_size, self.pg_hidden_channels
            )
            self.group_embedding = Embedding(
                self.phys_emb.group_size, self.pg_hidden_channels
            )

        assert (
            tag_hidden_channels + 2 * pg_hidden_channels + phys_hidden_channels
            < hidden_channels
        )

        # Main embedding
        self.embedding = Embedding(
            85,
            hidden_channels
            - tag_hidden_channels
            - self.phys_hidden_channels
            - 2 * self.pg_hidden_channels,
        )

        # Position encoding
        if self.use_positional_embeds:
            self.pe = PositionalEncoding(hidden_channels, 210)

        # Interaction block
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

        # Output block
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        # weigthed average & pooling
        if self.energy_head in {"pooling", "random"}:
            self.hierarchical_pooling = Hierarchical_Pooling(
                hidden_channels,
                self.act,
                NUM_POOLING_LAYERS,
                NUM_CLUSTERS,
                self.energy_head,
            )
        elif self.energy_head == "graclus":
            self.graclus = Graclus(hidden_channels, self.act)
        elif self.energy_head:
            self.w_lin = Linear(hidden_channels, 1)

        self.register_buffer("initial_atomref", atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        if self.use_mlp_phys:
            torch.nn.init.xavier_uniform_(self.phys_lin.weight)
        if self.use_tag:
            self.tag_embedding.reset_parameters()
        if self.use_pg:
            self.period_embedding.reset_parameters()
            self.group_embedding.reset_parameters()
        if self.energy_head in ["weighted-av-init-embeds", "weighted-av-final-embeds"]:
            self.w_lin.bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.w_lin.weight)
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z, pos, batch=None):
        """ """
        raise NotImplementedError

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"tag_hidden_channels={self.tag_hidden_channels}, "
            f"properties={self.phys_hidden_channels}, "
            f"period_hidden_channels={self.pg_hidden_channels}, "
            f"group_hidden_channels={self.pg_hidden_channels}, "
            f"energy_head={self.energy_head}",
            f"num_filters={self.num_filters}, "
            f"num_interactions={self.num_interactions}, "
            f"num_gaussians={self.num_gaussians}, "
            f"cutoff={self.cutoff})",
        )


@registry.register_model("new_schnet")
class NewSchNetWrap(NewSchNet):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        new_gnn,  # not used
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        hidden_channels=128,
        tag_hidden_channels=32,
        pg_hidden_channels=32,
        phys_hidden_channels=32,
        phys_embeds=False,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        readout="add",
        graph_rewiring=False,
        energy_head=False,
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.graph_rewiring = graph_rewiring

        super(NewSchNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            tag_hidden_channels=tag_hidden_channels,
            pg_hidden_channels=pg_hidden_channels,
            phys_hidden_channels=phys_hidden_channels,
            phys_embeds=phys_embeds,
            graph_rewiring=graph_rewiring,
            energy_head=energy_head,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        """"""
        # Re-compute on the fly the graph
        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        # Rewire the graph
        if not self.graph_rewiring:
            z = data.atomic_numbers.long()
            pos = data.pos
            batch = data.batch
        elif self.graph_rewiring == "remove-tag-0":
            data = remove_tag0_nodes(data)
            z = data.atomic_numbers.long()
            pos = data.pos
            batch = data.batch
        elif self.graph_rewiring == "one-supernode-per-graph":
            data = one_supernode_per_graph(data)
            z = data.atomic_numbers.long()
            pos = data.pos
            batch = data.batch
        elif self.graph_rewiring == "one-supernode-per-atom-type":
            data = one_supernode_per_atom_type(data)
            z = data.atomic_numbers.long()
            pos = data.pos
            batch = data.batch
        elif self.graph_rewiring == "one-supernode-per-atom-type-dist":
            data = one_supernode_per_atom_type_dist(data)
            z = data.atomic_numbers.long()
            pos = data.pos
            batch = data.batch
        else:
            raise ValueError(f"Unknown self.graph_rewiring {self.graph_rewiring}")

        # Use periodic boundary conditions
        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            edge_attr = self.distance_expansion(edge_weight)
        else:
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )
            # edge_index = data.edge_index
            row, col = edge_index
            edge_weight = (pos[row] - pos[col]).norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)

        h = self.embedding(z)

        if self.use_tag:
            assert data.tags is not None
            h_tag = self.tag_embedding(data.tags)
            h = torch.cat((h, h_tag), dim=1)

        if self.phys_emb.device != batch.device:
            self.phys_emb = self.phys_emb.to(batch.device)

        if self.use_phys_embeddings:
            h_phys = self.phys_emb.properties[z]
            if self.use_mlp_phys:
                h_phys = self.phys_lin(h_phys)
            h = torch.cat((h, h_phys), dim=1)

        if self.use_pg:
            # assert self.phys_emb.period is not None
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

        loss = None  # deal with pooling loss

        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        elif self.energy_head == "graclus":
            h, batch = self.graclus(h, edge_index, edge_weight, batch)

        elif self.energy_head:
            h, batch, loss = self.hierarchical_pooling(
                h, edge_index, edge_weight, batch
            )

        # MLP
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.energy_head in {"weigthed-av-final-embeds", "weigthed-av-final-embeds"}:
            h = h * alpha

        if self.atomref is not None:
            h = h + self.atomref(z)

        # Global pooling
        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.scale is not None:
            out = self.scale * out

        return out, loss

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy, pooling_loss = self._forward(data)

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
            return energy, pooling_loss

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
