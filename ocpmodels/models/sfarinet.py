""" Code of the Scalable Frame Averaging (Rotation Invariant) GNN
"""

import torch
from e3nn.o3 import spherical_harmonics
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad, get_pbc_distances
from ocpmodels.models.base_model import BaseModel
from ocpmodels.models.force_decoder import ForceDecoder
from ocpmodels.models.utils.pos_encodings import PositionalEncoding
from ocpmodels.models.utils.activations import swish
from ocpmodels.modules.phys_embeddings import PhysEmbedding


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EmbeddingBlock(nn.Module):
    def __init__(
        self,
        num_gaussians,
        hidden_channels,
        tag_hidden_channels,
        pg_hidden_channels,
        phys_hidden_channels,
        phys_embeds,
        graph_rewiring,
        act,
        edge_embed_type,
    ):
        super().__init__()
        self.act = act
        self.use_tag = tag_hidden_channels > 0
        self.use_pg = pg_hidden_channels > 0
        self.use_mlp_phys = phys_hidden_channels > 0 and phys_embeds
        self.use_positional_embeds = graph_rewiring in {
            "one-supernode-per-graph",
            "one-supernode-per-atom-type",
            "one-supernode-per-atom-type-dist",
        }
        self.edge_embed_type = edge_embed_type

        # Phys embeddings
        self.phys_emb = PhysEmbedding(
            props=phys_embeds, props_grad=phys_hidden_channels > 0, pg=self.use_pg
        )
        # With MLP
        if self.use_mlp_phys:
            self.phys_lin = Linear(self.phys_emb.n_properties, phys_hidden_channels)
        else:
            phys_hidden_channels = self.phys_emb.n_properties

        # Period + group embeddings
        if self.use_pg:
            self.period_embedding = Embedding(
                self.phys_emb.period_size, pg_hidden_channels
            )
            self.group_embedding = Embedding(
                self.phys_emb.group_size, pg_hidden_channels
            )

        # Tag embedding
        if tag_hidden_channels:
            self.tag_embedding = Embedding(3, tag_hidden_channels)

        # Positional encoding
        if self.use_positional_embeds:
            self.pe = PositionalEncoding(hidden_channels, 210)

        # Main embedding
        self.emb = Embedding(
            85,
            hidden_channels
            - tag_hidden_channels
            - phys_hidden_channels
            - 2 * pg_hidden_channels,
        )

        # MLP
        self.lin = Linear(hidden_channels, hidden_channels)

        # --- Edge embedding ---
        if self.edge_embed_type == "":
            self.lin_e = Linear(num_gaussians + 3, hidden_channels)
        elif self.edge_embed_type == "rij":
            self.lin_e = Linear(3, hidden_channels)
        elif self.edge_embed_type == "all_rij":
            self.lin_e = Linear(3, hidden_channels // 3)  # r_ij
            self.lin_e2 = Linear(3, hidden_channels // 3)  # norm r_ij
            self.lin_e3 = Linear(
                num_gaussians, hidden_channels - 2 * (hidden_channels // 3)
            )  # d_ij
        elif self.edge_embed_type == "sh":
            self.lin_e = Linear(15, hidden_channels)
        elif self.edge_embed_type == "all":
            self.lin_e = Linear(18, hidden_channels)
        else:
            raise ValueError("edge_embedding_type does not exist")

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        if self.use_mlp_phys:
            nn.init.xavier_uniform_(self.phys_lin.weight)
        if self.use_tag:
            self.tag_embedding.reset_parameters()
        if self.use_pg:
            self.period_embedding.reset_parameters()
            self.group_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e.weight)
        self.lin_e.bias.data.fill_(0)
        if self.edge_embed_type == "all_rij":
            nn.init.xavier_uniform_(self.lin_e2.weight)
            self.lin_e2.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_e3.weight)
            self.lin_e3.bias.data.fill_(0)

    def forward(
        self, z, rel_pos, edge_attr, tag=None, normalised_rel_pos=None, subnodes=None
    ):
        # --- Edge embedding --

        if self.edge_embed_type == "rij":
            e = self.lin_e(rel_pos)
        elif self.edge_embed_type == "all_rij":
            rel_pos = self.lin_e(rel_pos)  # r_ij
            normalized_rel_pos = self.lin_e2(normalised_rel_pos)  # norm r_ij
            edge_attr = self.lin_e3(edge_attr)  # d_ij
            e = torch.cat((rel_pos, edge_attr, normalized_rel_pos), dim=1)
        elif self.edge_embed_type == "sh":
            self.sh = spherical_harmonics(
                l=[1, 2, 3],
                x=normalised_rel_pos,
                normalize=False,
                normalization="component",
            )
            e = self.lin_e(self.sh)
        elif self.edge_embed_type == "all":
            self.sh = spherical_harmonics(
                l=[1, 2, 3],
                x=normalised_rel_pos,
                normalize=False,
                normalization="component",
            )
            e = torch.cat((rel_pos, self.sh), dim=1)
            e = self.lin_e(e)
        else:
            e = torch.cat((rel_pos, edge_attr), dim=1)
            e = self.lin_e(e)

        # --- Atom embedding --

        # Create atom embeddings based on its characteristic number
        h = self.emb(z)

        if self.phys_emb.device != h.device:
            self.phys_emb = self.phys_emb.to(h.device)

        # Concat tag embedding
        if self.use_tag:
            h_tag = self.tag_embedding(tag)
            h = torch.cat((h, h_tag), dim=1)

        # Concat physics embeddings
        if self.phys_emb.n_properties > 0:
            h_phys = self.phys_emb.properties[z]
            if self.use_mlp_phys:
                h_phys = self.phys_lin(h_phys)
            h = torch.cat((h, h_phys), dim=1)

        # Concat period & group embedding
        if self.use_pg:
            h_period = self.period_embedding(self.phys_emb.period[z])
            h_group = self.group_embedding(self.phys_emb.group[z])
            h = torch.cat((h, h_period, h_group), dim=1)

        # Add positional embedding
        if self.use_positional_embeds:
            idx_of_non_zero_val = (tag == 0).nonzero().T.squeeze(0)
            h_pos = torch.zeros_like(h, device=h.device)
            h_pos[idx_of_non_zero_val, :] = self.pe(subnodes).to(device=h_pos.device)
            h += h_pos

        # Apply MLP
        h = self.lin(h)

        return h, e


class InteractionBlock(MessagePassing):
    def __init__(self, hidden_channels, act):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, h, edge_index, e):
        h = self.propagate(edge_index, x=h, W=e)
        h = self.lin(self.act(h))
        return h

    def message(self, x_j, W):
        return x_j * W


class OutputBlock(nn.Module):
    def __init__(self, energy_head, hidden_channels, act):
        super().__init__()
        self.energy_head = energy_head
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)

        # weighted average
        if self.energy_head == "weighted-av-final-embeds":
            self.w_lin = Linear(hidden_channels, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.energy_head == "weighted-av-final-embeds":
            nn.init.xavier_uniform_(self.w_lin.weight)
            self.w_lin.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, batch, alpha):
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

        # Global pooling
        out = scatter(h, batch, dim=0, reduce="add")

        return out


@registry.register_model("sfarinet")
class SfariNet(BaseModel):
    r"""The Scalable Frame Averaging Rotation Invariant GNN model Sfarinet.

    Args:
        cutoff (float): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        use_pbc (bool): Use of periodic boundary conditions.
            (default: true)
        max_num_neighbors (int): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        graph_rewiring (str): Method used to create the graph,
            among "", remove-tag-0, supernodes.
        energy_head (str): Method to compute energy prediction
            from atom representations.
        hidden_channels (int): Hidden embedding size.
            (default: :obj:`128`)
        tag_hidden_channels (int): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int): Hidden period and group embed size.
            (default: obj:`32`)
        phys_embed (bool): Concat fixed physics-aware embeddings.
        phys_hidden_channels (int): Hidden size of learnable phys embed.
            (default: obj:`32`)
        num_interactions (int): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        force_decoder_type (str): Type of force decoder to use.
        force_decoder_model_config (dict): Dictionary of config parameters
            for the decoder's model
        edge_embed_type (str): type of edge_embedding
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.cutoff = kwargs["cutoff"]
        self.use_pbc = kwargs["use_pbc"]
        self.max_num_neighbors = kwargs["max_num_neighbors"]
        self.regress_forces = kwargs["regress_forces"]
        self.energy_head = kwargs["energy_head"]
        self.edge_embed_type = kwargs["edge_embed_type"]

        self.distance_expansion = GaussianSmearing(
            0.0, self.cutoff, kwargs["num_gaussians"]
        )
        self.act = (
            getattr(nn.functional, kwargs["act"]) if kwargs["act"] != "swish" else swish
        )

        # Embedding block
        self.embed_block = EmbeddingBlock(
            kwargs["num_gaussians"],
            kwargs["hidden_channels"],
            kwargs["tag_hidden_channels"],
            kwargs["pg_hidden_channels"],
            kwargs["phys_hidden_channels"],
            kwargs["phys_embeds"],
            kwargs["graph_rewiring"],
            self.act,
            kwargs["edge_embed_type"],
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    kwargs["hidden_channels"],
                    self.act,
                )
                for _ in range(kwargs["num_interactions"])
            ]
        )

        # Output block
        self.output_block = OutputBlock(
            self.energy_head, kwargs["hidden_channels"], self.act
        )

        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin = Linear(kwargs["hidden_channels"], 1)

        if not self.regress_forces and kwargs["force_decoder_type"]:
            print(
                "\nWarning: force_decoder_type is set to",
                kwargs["force_decoder_type"],
                "but regress_forces is False. Ignoring force_decoder_type.\n",
            )

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

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        return self.decoder(preds["hidden_state"])

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
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
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            rel_pos = out["distance_vec"]
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
            rel_pos = pos[row] - pos[col]
            edge_weight = rel_pos.norm(dim=-1)
            # edge_weight = data.distances
            edge_attr = self.distance_expansion(edge_weight)

        # Normalize and squash to [0,1] for gaussian basis
        rel_pos_normalized = None
        if self.edge_embed_type in {"sh", "all_rij", "all"}:
            rel_pos_normalized = (rel_pos / edge_weight.view(-1, 1) + 1) / 2.0

        # Embedding block
        h, e = self.embed_block(z, rel_pos, edge_attr, data.tags, rel_pos_normalized)

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)
        else:
            alpha = None

        # Interaction blocks
        for interaction in self.interaction_blocks:
            h = h + interaction(h, edge_index, e)
            # potential output block for skip connection

        # Output block
        energy = self.output_block(h, edge_index, edge_weight, batch, alpha)
        # or skip-connection

        # Force-head for S2EF, IS2RS

        preds = {"energy": energy, "hidden_state": h}

        return preds
