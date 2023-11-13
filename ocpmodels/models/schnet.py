"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from math import pi as PI

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.base_model import BaseModel
from ocpmodels.models.force_decoder import ForceDecoder
from ocpmodels.models.utils.pos_encodings import PositionalEncoding
from ocpmodels.modules.phys_embeddings import PhysEmbedding


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


@registry.register_model("schnet")
class SchNet(BaseModel):
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
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        use_pbc (bool, optional): Use of periodic boundary conditions.
            (default: true)
        otf_graph (bool, optional): Recompute radius graph.
            (default: false)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        graph_rewiring (str, optional): Method used to create the graph,
            among "", remove-tag-0, supernodes.
        energy_head (str, optional): Method to compute energy prediction
            from atom representations.
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        tag_hidden_channels (int, optional): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int, optional): Hidden period and group embed size.
            (default: obj:`32`)
        phys_embed (bool, optional): Concat fixed physics-aware embeddings.
        phys_hidden_channels (int, optional): Hidden size of learnable phys embed.
            (default: obj:`32`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
        force_decoder_model_config (dict): config of the force decoder model.
            keys: "model_type", "hidden_channels", "num_layers", "num_heads",
        force_decoder_type (str): type of the force decoder model.
            (options: "mlp", "simple", "res", "res_updown")
    """

    url = "http://www.quantum-machine.org/datasets/trained_schnet_models.zip"

    def __init__(self, **kwargs):
        super().__init__()

        import ase

        self.use_pbc = kwargs["use_pbc"]
        self.cutoff = kwargs["cutoff"]
        self.otf_graph = kwargs["otf_graph"]
        self.scale = None
        self.regress_forces = kwargs["regress_forces"]

        self.num_filters = kwargs["num_filters"]
        self.num_interactions = kwargs["num_interactions"]
        self.num_gaussians = kwargs["num_gaussians"]
        self.max_num_neighbors = kwargs["max_num_neighbors"]
        self.readout = kwargs["readout"]
        self.hidden_channels = kwargs["hidden_channels"]
        self.tag_hidden_channels = kwargs["tag_hidden_channels"]
        self.use_tag = self.tag_hidden_channels > 0
        self.pg_hidden_channels = kwargs["pg_hidden_channels"]
        self.use_pg = self.pg_hidden_channels > 0
        self.phys_hidden_channels = kwargs["phys_hidden_channels"]
        self.energy_head = kwargs["energy_head"]
        self.use_phys_embeddings = kwargs["phys_embeds"]
        self.use_mlp_phys = self.phys_hidden_channels > 0 and kwargs["phys_embeds"]
        self.use_positional_embeds = kwargs["graph_rewiring"] in {
            "one-supernode-per-graph",
            "one-supernode-per-atom-type",
            "one-supernode-per-atom-type-dist",
        }

        self.register_buffer(
            "initial_atomref",
            torch.tensor(kwargs["atomref"]) if kwargs["atomref"] is not None else None,
        )
        self.atomref = None
        if kwargs["atomref"] is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(torch.tensor(kwargs["atomref"]))

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        # self.covalent_radii = torch.from_numpy(ase.data.covalent_radii)
        # self.vdw_radii = torch.from_numpy(ase.data.vdw_radii)
        self.register_buffer("atomic_mass", atomic_mass)

        if self.use_tag:
            self.tag_embedding = Embedding(3, self.tag_hidden_channels)

        # Phys embeddings
        self.phys_emb = PhysEmbedding(props=kwargs["phys_embeds"], pg=self.use_pg)
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
            self.tag_hidden_channels
            + 2 * self.pg_hidden_channels
            + self.phys_hidden_channels
            < self.hidden_channels
        )

        # Main embedding
        self.embedding = Embedding(
            85,
            self.hidden_channels
            - self.tag_hidden_channels
            - self.phys_hidden_channels
            - 2 * self.pg_hidden_channels,
        )

        # Position encoding
        if self.use_positional_embeds:
            self.pe = PositionalEncoding(self.hidden_channels, 210)

        # Interaction block
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)
        self.interactions = ModuleList()
        for _ in range(self.num_interactions):
            block = InteractionBlock(
                self.hidden_channels, self.num_gaussians, self.num_filters, self.cutoff
            )
            self.interactions.append(block)

        # Output block
        self.lin1 = Linear(self.hidden_channels, self.hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(self.hidden_channels // 2, 1)

        # weighted average & pooling
        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            self.w_lin = Linear(self.hidden_channels, 1)

        # Force head
        self.decoder = (
            ForceDecoder(
                kwargs["force_decoder_type"],
                kwargs["hidden_channels"],
                kwargs["force_decoder_model_config"],
                self.act,
            )
            if "direct" in self.regress_forces
            else None
        )

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
        if self.energy_head in {"weighted-av-init-embeds", "weighted-av-final-embeds"}:
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

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        return self.decoder(preds["hidden_state"])

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
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
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

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

        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        hidden_state = h  # store hidden rep for force head

        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        # MLP
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        if self.atomref is not None:
            h = h + self.atomref(z)

        # Global pooling
        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.scale is not None:
            out = self.scale * out

        return {
            "energy": out,
            "hidden_state": hidden_state,
        }
