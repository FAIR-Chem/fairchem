""" Code of the Scalable Frame Averaging (Rotation Invariant) GNN
"""
import torch
import torch.nn.functional as F
from e3nn.o3 import spherical_harmonics
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn import MessagePassing, TransformerConv, radius_graph
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import dropout_edge
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad, get_pbc_distances
from ocpmodels.models.base_model import BaseModel
from ocpmodels.models.force_decoder import ForceDecoder
from ocpmodels.models.utils.activations import swish
from ocpmodels.models.utils.attention_model import TransfoAttConv
from ocpmodels.models.utils.pos_encodings import PositionalEncoding
from ocpmodels.modules.phys_embeddings import PhysEmbedding
from ocpmodels.modules.pooling import Graclus, Hierarchical_Pooling

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
        elif self.edge_embed_type == "all_rij":
            self.lin_e1 = Linear(3, num_filters // 2)  # r_ij
            self.lin_e12 = Linear(
                num_gaussians, num_filters - (num_filters // 2)
            )  # d_ij
        elif self.edge_embed_type == "sh":
            self.lin_e1 = Linear(15, num_filters)
        elif self.edge_embed_type == "all":
            self.lin_e1 = Linear(18 + num_gaussians, num_filters)
        else:
            raise ValueError("edge_embedding_type does not exist")

        if self.second_layer_MLP:
            self.lin_e2 = Linear(num_filters, num_filters)

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
        nn.init.xavier_uniform_(self.lin_e1.weight)
        self.lin_e1.bias.data.fill_(0)
        if self.second_layer_MLP:
            nn.init.xavier_uniform_(self.lin_2.weight)
            self.lin_2.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_e2.weight)
            self.lin_e2.bias.data.fill_(0)
        if self.edge_embed_type == "all_rij":
            nn.init.xavier_uniform_(self.lin_e12.weight)
            self.lin_e12.bias.data.fill_(0)

    def forward(
        self, z, rel_pos, edge_attr, tag=None, normalised_rel_pos=None, subnodes=None
    ):
        # --- Edge embedding --

        if self.edge_embed_type == "rij":
            e = self.lin_e1(rel_pos)
        elif self.edge_embed_type == "all_rij":
            rel_pos = self.lin_e1(rel_pos)  # r_ij
            edge_attr = self.lin_e12(edge_attr)  # d_ij
            e = torch.cat((rel_pos, edge_attr), dim=1)
        elif self.edge_embed_type == "sh":
            self.sh = spherical_harmonics(
                l=[1, 2, 3],
                x=normalised_rel_pos,
                normalize=False,
                normalization="component",
            )
            e = self.lin_e1(self.sh)
        elif self.edge_embed_type == "all":
            self.sh = spherical_harmonics(
                l=[1, 2, 3],
                x=normalised_rel_pos,
                normalize=False,
                normalization="component",
            )
            e = torch.cat((rel_pos, self.sh, edge_attr), dim=1)
            e = self.lin_e1(e)

        e = self.act(e)  # can comment out

        if self.second_layer_MLP:
            # e = self.lin_e2(e)
            e = self.act(self.lin_e2(e))

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
        h = self.act(self.lin(h))
        if self.second_layer_MLP:
            h = self.act(self.lin_2(h))

        return h, e


class InteractionBlock(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_filters,
        act,
        mp_type,
        complex_mp,
        att_heads,
        graph_norm,
        dropout_lin,
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.mp_type = mp_type
        self.hidden_channels = hidden_channels
        self.complex_mp = complex_mp
        self.graph_norm = graph_norm
        self.dropout_lin = float(dropout_lin)
        if graph_norm:
            self.graph_norm = GraphNorm(
                hidden_channels if "updown" not in self.mp_type else num_filters
            )

        if self.mp_type == "simple":
            self.lin_geom = nn.Linear(num_filters, hidden_channels)
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        elif self.mp_type == "sfarinet":
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        elif self.mp_type == "updownscale":
            self.lin_geom = nn.Linear(num_filters, num_filters)  # like 'simple'
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updownscale_base":
            self.lin_geom = nn.Linear(num_filters + 2 * hidden_channels, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "base_with_att":
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)
            # self.lin_geom = AttConv(hidden_channels, heads=1, concat=True, bias=True)
            self.lin_geom = TransfoAttConv(
                hidden_channels,
                hidden_channels,
                heads=att_heads,
                concat=False,
                root_weight=False,
                edge_dim=num_filters,
            )
        elif self.mp_type == "att":
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)
            self.lin_geom = TransformerConv(
                hidden_channels,
                hidden_channels,
                heads=att_heads,
                concat=False,
                root_weight=False,
                edge_dim=num_filters,
            )

        elif self.mp_type == "local_env":
            self.lin_geom = nn.Linear(num_filters, hidden_channels)
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        elif self.mp_type == "updown_local_env":
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_up = nn.Linear(2 * num_filters, hidden_channels)

        else:  # base
            self.lin_geom = nn.Linear(
                num_filters + 2 * hidden_channels, hidden_channels
            )
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        if self.complex_mp:
            self.other_mlp = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.mp_type not in {"sfarinet", "att", "base_with_att"}:
            nn.init.xavier_uniform_(self.lin_geom.weight)
            self.lin_geom.bias.data.fill_(0)
        if self.complex_mp:
            nn.init.xavier_uniform_(self.other_mlp.weight)
            self.other_mlp.bias.data.fill_(0)
        if self.mp_type in {"updownscale", "updownscale_base", "updown_local_env"}:
            nn.init.xavier_uniform_(self.lin_up.weight)
            self.lin_up.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_down.weight)
            self.lin_down.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.lin_h.weight)
            self.lin_h.bias.data.fill_(0)

    def forward(self, h, edge_index, e):
        # Define edge embedding

        if self.dropout_lin > 0:
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )

        if self.mp_type in {"base", "updownscale_base"}:
            e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)

        if self.mp_type in {
            "simple",
            "updownscale",
            "base",
            "updownscale_base",
            "local_env",
        }:
            e = self.act(self.lin_geom(e))  # TODO: remove act() ?

        # --- Message Passing block --

        if self.mp_type == "updownscale" or self.mp_type == "updownscale_base":
            h = self.act(self.lin_down(h))  # downscale node rep.
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = self.act(self.lin_up(h))  # upscale node rep.

        elif self.mp_type == "att":
            h = self.lin_geom(h, edge_index, edge_attr=e)
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = self.act(self.lin_h(h))

        elif self.mp_type == "base_with_att":
            h = self.lin_geom(h, edge_index, edge_attr=e)  # propagate is inside
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = self.act(self.lin_h(h))

        elif self.mp_type == "local_env":
            chi = self.propagate(edge_index, x=h, W=e, local_env=True)
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            h = h + chi
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = h = self.act(self.lin_h(h))

        elif self.mp_type == "updown_local_env":
            h = self.act(self.lin_down(h))
            chi = self.propagate(edge_index, x=h, W=e, local_env=True)
            e = F.dropout(
                e, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            e = self.lin_geom(e)
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = torch.cat((h, chi), dim=1)
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = self.lin_up(h)

        elif self.mp_type in {"base", "simple", "sfarinet"}:
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = self.act(self.lin_h(h))

        else:
            raise ValueError("mp_type provided does not exist")

        if self.complex_mp:
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = self.act(self.other_mlp(h))

        return h

    def message(self, x_j, W, local_env=None):
        if local_env is not None:
            return W
        else:
            return x_j * W


class OutputBlock(nn.Module):
    def __init__(self, energy_head, hidden_channels, act, dropout_lin):
        super().__init__()
        self.energy_head = energy_head
        self.act = act
        self.dropout_lin = float(dropout_lin)

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
        if hasattr(self, "deup_lin"):
            nn.init.xavier_uniform_(self.deup_lin.weight)
            self.deup_lin.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, batch, alpha, data=None):
        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        elif self.energy_head == "graclus":
            h, batch = self.graclus(h, edge_index, edge_weight, batch)

        elif self.energy_head in {"pooling", "random"}:
            h, batch, pooling_loss = self.hierarchical_pooling(
                h, edge_index, edge_weight, batch
            )

        # MLP
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training or self.deup_inference
        )
        h = self.lin1(h)
        h = self.act(h)
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training or self.deup_inference
        )
        h = self.lin2(h)

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        # Global pooling
        out = scatter(h, batch, dim=0, reduce="add")

        return out


@registry.register_model("faenet")
class FAENet(BaseModel):
    r"""Frame Averaging GNN model FAENet.

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
        second_layer_MLP (bool): use 2-layers MLP at the end of the Embedding block.
        skip_co (str): add a skip connection between each interaction block and
            energy-head. ("add", False, "concat", "concat_atom")
        edge_embed_type (str, in {'rij','all_rij','sh', 'all'}): input feature
            of the edge embedding block.
        edge_embed_hidden (int): size of edge representation.
            could be num_filters or hidden_channels.
        mp_type (str, in {'base', 'simple', 'updownscale', 'att', 'base_with_att', 'local_env'
            'updownscale_base', 'updownscale', 'updown_local_env', 'sfarinet'}}):
            specificies the MP of the interaction block.
        graph_norm (bool): whether to apply batch norm after every linear layer.
        complex_mp (bool): whether to add a second layer MLP at the end of each Interaction
        force_decoder_model_config (dict): config of the force decoder model.
            keys: "model_type", "hidden_channels", "num_layers", "num_heads",
        force_decoder_type (str): type of the force decoder model.
            (options: "mlp", "simple", "res", "res_updown")
        dropout_lin (float): dropout rate for linear layers.
        dropout_edge (float): dropout rate for edges.
        dropout_lowest_layer (str): The first layer to apply dropout to. All subsequent
            layers will have the same dropout rate. Can be `inter-{i}` or `output`.
            Defaults to `output` if none is provided.
        first_trainable_layer (str): lowest layer to NOT freeze. All previous layers will be frozen.
            Can be ``, `embed`, `inter-{i}`, `output`, or `dropout`. If it is `` then no layer is frozen.
            If it is `dropout` then it will be set to the layer before `dropout_lowest_layer`. 
            Defaults to ``.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.cutoff = kwargs["cutoff"]
        self.energy_head = kwargs["energy_head"]
        self.regress_forces = kwargs["regress_forces"]
        self.use_pbc = kwargs["use_pbc"]
        self.max_num_neighbors = kwargs["max_num_neighbors"]
        self.edge_embed_type = kwargs["edge_embed_type"]
        self.skip_co = kwargs["skip_co"]
        self.dropout_edge = float(kwargs.get("dropout_edge") or 0)
        self.dropout_lin = float(kwargs.get("dropout_lin") or 0)
        self.dropout_lowest_layer = kwargs.get("dropout_lowest_layer", "output") or ""
        self.first_trainable_layer = kwargs.get("first_trainable_layer", "") or ""

        if kwargs["mp_type"] == "sfarinet":
            kwargs["num_filters"] = kwargs["hidden_channels"]

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
                    kwargs["complex_mp"],
                    kwargs["att_heads"],
                    kwargs["graph_norm"],
                    (
                        print(
                            "🗑️ Setting dropout_lin for interaction block ",
                            f"{i} / {kwargs['num_interactions']}",
                        )
                        or self.dropout_lin
                    )
                    if "inter" in self.dropout_lowest_layer
                    and (i >= int(self.dropout_lowest_layer.split("-")[-1]))
                    else 0,
                )
                for i in range(kwargs["num_interactions"])
            ]
        )

        # Output block
        self.output_block = OutputBlock(
            self.energy_head,
            kwargs["hidden_channels"],
            self.act,
            (print("🗑️ Setting dropout_lin for output block") or self.dropout_lin)
            if (
                "inter" in self.dropout_lowest_layer
                or "output" in self.dropout_lowest_layer
            )
            else 0,
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

        # Skip co
        if self.skip_co == "concat":
            self.mlp_skip_co = Linear((kwargs["num_interactions"] + 1), 1)
        elif self.skip_co == "concat_atom":
            self.mlp_skip_co = Linear(
                ((kwargs["num_interactions"] + 1) * kwargs["hidden_channels"]),
                kwargs["hidden_channels"],
            )

        self.freeze(self.first_trainable_layer)
        print()

    def freeze(self, lowest_layer="dropout"):
        assert (
            not lowest_layer
            or "inter_" in lowest_layer
            or lowest_layer == "output"
            or lowest_layer == "dropout"
            or lowest_layer == "none"
        ), (
            "first_trainable_layer must be None, '', 'inter_{i}', "
            + f"'output', 'dropout', or 'none'. Received: {lowest_layer}"
        )
        if lowest_layer == "dropout":
            lowest_layer = self.dropout_lowest_layer

        if not lowest_layer:
            print("⛄️ No layer to freeze")
            return

        if lowest_layer == "embed":
            print("⛄️ No layer to freeze")
            return

        print("⛄️ Freezing embedding layer")
        self.freeze_layer(self.embed_block)
        if lowest_layer == "inter_0":
            return

        interaction_block_idx = (
            int(lowest_layer.replace("inter_", ""))
            if "inter_" in lowest_layer
            else len(self.interaction_blocks)
        )
        if interaction_block_idx < 0:
            interaction_block_idx = len(self.interaction_blocks) + interaction_block_idx
            self.first_trainable_layer = f"inter_{interaction_block_idx}"

        for ib in range(interaction_block_idx):
            if ib >= len(self.interaction_blocks):
                print("⛄️ Trying to freeze too many interaction blocks")
                break
            self.freeze_layer(self.interaction_blocks[ib])

        if ib > 0:
            print(
                "⛄️ Freezing interaction blocks 0 ->",
                f"{ib} / {len(self.interaction_blocks)}",
            )
        else:
            print(f"⛄️ Freezing interaction block 0 / {len(self.interaction_blocks)}")

        if "output" in lowest_layer:
            return

        print("⛄️ Freezing output block")
        self.freeze_layer(self.output_block)

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

            if self.dropout_edge > 0:
                edge_index, edge_mask = dropout_edge(
                    data.edge_index,
                    p=self.dropout_edge,
                    force_undirected=True,
                    training=self.training or self.deup_inference,
                )

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
            if self.dropout_edge > 0:
                edge_index, edge_mask = dropout_edge(
                    edge_index,
                    p=self.dropout_edge,
                    force_undirected=True,
                    training=self.training or self.deup_inference,
                )
                edge_weight = edge_weight[edge_mask]
                edge_attr = edge_attr[edge_mask]
                rel_pos = rel_pos[edge_mask]

        # Normalize and squash to [0,1] for gaussian basis
        rel_pos_normalized = None
        if self.edge_embed_type in {"sh", "all_rij", "all"}:
            rel_pos_normalized = (rel_pos / edge_weight.view(-1, 1) + 1) / 2.0

        pooling_loss = None  # deal with pooling loss
        q = None  # hidden state for deup dataset

        # Embedding block
        h, e = self.embed_block(z, rel_pos, edge_attr, data.tags, rel_pos_normalized)
        if "inter" and "0" in self.first_trainable_layer:
            q = h.clone().detach()
        # print("h, e: ", h.mean(), e.mean())

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)
        else:
            alpha = None

        # Interaction blocks
        energy_skip_co = []
        for ib, interaction in enumerate(self.interaction_blocks):
            if self.skip_co == "concat_atom":
                energy_skip_co.append(h)
            elif self.skip_co:
                energy_skip_co.append(
                    self.output_block(h, edge_index, edge_weight, batch, alpha, data)
                )
            if "inter" in self.first_trainable_layer and ib == int(
                self.first_trainable_layer.split("_")[1]
            ):
                q = h.clone().detach()
            h = h + interaction(h, edge_index, e)
            # print("h: ", h.mean())

        # Atom skip-co
        if self.skip_co == "concat_atom":
            energy_skip_co.append(h)
            h = self.act(self.mlp_skip_co(torch.cat(energy_skip_co, dim=1)))

        if "output" in self.first_trainable_layer:
            q = h.clone().detach()

        energy = self.output_block(h, edge_index, edge_weight, batch, alpha, data=data)
        # print("energy: ", energy.mean())

        # Skip-connection
        energy_skip_co.append(energy)
        if self.skip_co == "concat":
            energy = self.mlp_skip_co(torch.cat(energy_skip_co, dim=1))
        elif self.skip_co == "add":
            energy = sum(energy_skip_co)

        preds = {
            "energy": energy,
            "pooling_loss": pooling_loss,
            "hidden_state": h,
            "q": q,
        }

        return preds
