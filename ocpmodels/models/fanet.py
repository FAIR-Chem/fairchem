""" Code of the Scalable Frame Averaging (Rotation Invariant) GNN
"""
import math

import torch
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad, get_pbc_distances
from ocpmodels.models.base_model import BaseModel
from ocpmodels.models.force_decoder import ForceDecoder
from ocpmodels.models.utils.pos_encodings import PositionalEncoding
from ocpmodels.modules.phys_embeddings import PhysEmbedding
from ocpmodels.modules.pooling import Graclus, Hierarchical_Pooling

try:
    from torch_geometric.nn.acts import swish
except ImportError:
    from torch_geometric.nn.resolver import swish

NUM_CLUSTERS = 20
NUM_POOLING_LAYERS = 1


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
        num_filters,
        hidden_channels,
        tag_hidden_channels,
        pg_hidden_channels,
        phys_hidden_channels,
        phys_embeds,
        graph_rewiring,
        act,
        second_layer_MLP,
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
        self.second_layer_MLP = second_layer_MLP
        self.edge_embed_type = edge_embed_type

        # --- Node embedding ---

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
        if self.second_layer_MLP:
            self.lin_2 = Linear(hidden_channels, hidden_channels)

        # --- Edge embedding ---

        # TODO: change some num_filters to edge_embed_hidden
        if self.edge_embed_type == "rij":
            self.lin_e1 = Linear(3, num_filters)
            self.lin_e2 = Linear(num_filters, num_filters)
        elif self.edge_embed_type == "all-rij":
            self.lin_e1 = Linear(3, num_filters // 3)  # r_ij
            self.lin_e12 = Linear(3, num_filters // 3)  # norm r_ij
            self.lin_e13 = Linear(num_gaussians, math.ceil(num_filters // 3))  # d_ij
            self.lin_e2 = Linear(num_filters, num_filters)  # mlp of concat
        elif self.edge_embed_type == "sh":
            self.spherical_harmonics = None
            # TODO: compute SH e3nn.o3.spherical_harmonics,
            # order 3, concatenated.
            self.lin_e1 = Linear(15, num_filters)
            self.lin_e2 = Linear(num_filters, num_filters)
        elif self.edge_embed_type == "all":
            self.lin_e1 = Linear(18, num_filters)
            self.lin_e2 = Linear(num_filters, num_filters)
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
        if self.second_layer_MLP:
            nn.init.xavier_uniform_(self.lin_2.weight)
            self.lin_2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e1.weight)
        self.lin_e1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e2.weight)
        self.lin_e2.bias.data.fill_(0)
        if self.edge_embed_type == "all_rij":
            nn.init.xavier_uniform_(self.lin_e12.weight)
            self.lin_e12.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_e13.weight)
            self.lin_e13.bias.data.fill_(0)

    def forward(
        self, z, rel_pos, edge_attr, tag=None, normalised_rel_pos=None, subnodes=None
    ):

        # --- Edge embedding --

        if self.edge_embed_type == "rij":
            e = self.act(self.lin_e1(rel_pos))
            e = self.lin_e2(e)
        elif self.edge_embed_type == "all_rij":
            rel_pos = self.act(self.lin_e1(rel_pos))  # r_ij
            normalized_rel_pos = self.act(self.lin_e12(normalised_rel_pos))  # norm r_ij
            edge_attr = self.act(self.lin_e13(edge_attr))  # d_ij
            e = torch.cat((rel_pos, edge_attr, normalized_rel_pos), dim=1)
            e = self.lin_e2(e)
        elif self.edge_embed_type == "sh":
            e = self.act(self.lin_e1(self.spherical_harmonics))
            e = self.lin_e2(e)
        elif self.edge_embed_type == "all":
            e = torch.cat((rel_pos, self.spherical_harmonics), dim=1)
            e = self.act(self.lin_e1(e))
            e = self.lin_e2(e)

        # --- Node embedding --

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

        # MLP
        h = self.lin(h)
        if self.second_layer_MLP:
            h = self.lin_2(self.act(h))

        return h, e


class InteractionBlock(MessagePassing):
    def __init__(self, hidden_channels, num_filters, act, mp_type):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.mp_type = mp_type

        if self.mp_type == "base_with_att":
            # --- Compute attention coefficients if required --
            # Change message function
            pass

        if self.mp_type == "att":
            # --- Compute attention coefficients if required --
            # Change message function
            pass

        if self.mp_type == "local_env":
            pass

        elif self.mp_type == "updownscale":
            self.lin_geom = nn.Linear(
                num_filters + 2 * hidden_channels, hidden_channels
            )
            # self.lin_geom = nn.Linear(num_filters, num_filters)  # like 'simple'
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "simple":
            self.lin_geom = nn.Linear(num_filters, hidden_channels)
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        else:  # base
            self.lin_geom = nn.Linear(
                num_filters + 2 * hidden_channels, hidden_channels
            )
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_geom.weight)
        self.lin_geom.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_h.weight)
        self.lin_h.bias.data.fill_(0)
        if self.mp_type == "updownscale":
            nn.init.xavier_uniform_(self.lin_up.weight)
            self.lin_up.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_down.weight)
            self.lin_down.bias.data.fill_(0)

    def forward(self, h, edge_index, e):

        if self.mp_type != "simple":
            e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)

        # W = self.lin_e_2(self.act(self.lin_e_1(e)))  # transform edge rep
        W = self.lin_geom(e)

        if self.mp_type == "updownscale":
            h = self.lin_down(h)  # downscale node rep.
            h = self.propagate(edge_index, x=h, W=W)  # propagate
            h = self.lin_up(self.act(h))  # upscale node rep.
        elif self.mp_type == "att":
            # Look at So3krates code
            pass
        elif self.mp_type == "base_with_att":
            # Combine above and base
            pass
        elif self.mp_type == "local_env":
            pass
        else:
            h = self.lin_h(self.act(h))
            h = self.propagate(edge_index, x=h, W=W)  # propagate

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

        # weighted average & pooling
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
        elif self.energy_head == "weighted-av-final-embeds":
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

        elif self.energy_head == "graclus":
            h, batch = self.graclus(h, edge_index, edge_weight, batch)

        elif self.energy_head in {"pooling", "random"}:
            h, batch, pooling_loss = self.hierarchical_pooling(
                h, edge_index, edge_weight, batch
            )

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


@registry.register_model("fanet")
class FANet(BaseModel):
    r"""Frame Averaging GNN model FANet.

    Args:
        cutoff (float): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        use_pbc (bool): Use of periodic boundary conditions.
            (default: true)
        act (str): activation function
            (default: swish)
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
            (default: :obj:`4`)
        num_gaussians (int): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        second_layer_MLP (bool): use 2-layers MLP at the end of embedding block.
        skip_co (bool): add a skip connection between interaction blocks and
            energy-head.
        edge_embed_type (str, in {'rij','all_rij','sh', 'all'}): input feature
            of the edge embedding block.
        edge_embed_hidden (int): size of edge representation.
            could be num_filters or hidden_channels.
        mp_type (str, in {'base', 'simple', 'updownscale', 'att', 'base_with_att', 'local_env}):
            specificies the MP of the interaction block.
    """

    def __init__(self, **kwargs):

        super(FANet, self).__init__()

        self.cutoff = kwargs["cutoff"]
        self.energy_head = kwargs["energy_head"]
        self.regress_forces = kwargs["regress_forces"]
        self.use_pbc = kwargs["use_pbc"]
        self.max_num_neighbors = kwargs["max_num_neighbors"]
        self.edge_embed_type = kwargs["edge_embed_type"]
        self.skip_co = kwargs["skip_co"]

        self.act = (
            getattr(nn.functional, kwargs["act"]) if kwargs["act"] != "swish" else swish
        )
        self.use_positional_embeds = kwargs["graph_rewiring"] in {
            "one-supernode-per-graph",
            "one-supernode-per-atom-type",
            "one-supernode-per-atom-type-dist",
        }
        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(
            0.0, self.cutoff, kwargs["num_gaussians"]
        )

        # Embedding block
        self.embed_block = EmbeddingBlock(
            kwargs["num_gaussians"],
            kwargs["num_filters"],
            kwargs["hidden_channels"],
            kwargs["tag_hidden_channels"],
            kwargs["pg_hidden_channels"],
            kwargs["phys_hidden_channels"],
            kwargs["phys_embeds"],
            kwargs["graph_rewiring"],
            self.act,
            kwargs["second_layer_MLP"],
            kwargs["edge_embed_type"],
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    kwargs["hidden_channels"],
                    kwargs["num_filters"],
                    self.act,
                    kwargs["mp_type"],
                )
                for _ in range(kwargs["num_interactions"])
            ]
        )

        # Output block
        self.output_block = OutputBlock(
            self.energy_head, kwargs["hidden_channels"], self.act
        )

        # Energy head
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin = Linear(kwargs["hidden_channels"], 1)

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
            edge_attr = self.distance_expansion(edge_weight)

        # Normalize and squash to [0,1] for gaussian basis
        rel_pos_normalized = None
        if self.edge_embed_type == "all_rij":
            rel_pos_normalized = (rel_pos / edge_weight.view(-1, 1) + 1) / 2.0

        pooling_loss = None  # deal with pooling loss

        # Embedding block
        h, e = self.embed_block(z, rel_pos, edge_attr, data.tags, rel_pos_normalized)

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)
        else:
            alpha = None
        energy_skip_co = torch.zeros(max(batch) + 1, device=h.device).unsqueeze(1)

        # Interaction blocks
        for interaction in self.interaction_blocks:
            if self.skip_co:
                energy_skip_co += self.output_block(
                    h, edge_index, edge_weight, batch, alpha
                )
            h = h + interaction(h, edge_index, e)

        # Output block
        energy = self.output_block(h, edge_index, edge_weight, batch, alpha)
        # skip-connection
        if self.skip_co:
            energy += energy_skip_co

        preds = {"energy": energy, "pooling_loss": pooling_loss, "hidden_state": h}

        return preds
