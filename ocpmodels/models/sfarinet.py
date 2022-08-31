""" Code of the Scalable Frame Averaging (Rotation Invariant) GNN
"""

from math import sqrt

import torch
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.nn.acts import swish
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import get_pbc_distances
from ocpmodels.models.base import BaseModel
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

        # Phys embeddings
        self.phys_emb = PhysEmbedding(props=phys_embeds, props_grad=phys_hidden_channels > 0, pg=self.use_pg)
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
        self.lin_e = Linear(num_gaussians + 3, hidden_channels)
        # TODO: check if it has to be hidden_channels

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

    def forward(self, z, rel_pos, edge_attr, tag=None, subnodes=None):
        # Create edge embeddings from d_ij || r_ij
        e = torch.cat((rel_pos, edge_attr), dim=1)
        # Extension: learn a bond feature vector and concat to above

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
        e = self.lin_e(e)

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
        elif self.energy_head == "weigthed-av-final-embeds":
            self.w_lin = Linear(hidden_channels, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.energy_head == "weigthed-av-final-embeds":
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

        if self.energy_head in {"weigthed-av-initial-embeds", "weigthed-av-final-embeds"}:
            h = h * alpha

        # Global pooling
        out = scatter(h, batch, dim=0, reduce="add")

        return out


@registry.register_model("sfarinet")
class SfariNet(BaseModel):
    def __init__(
        self,
        num_atoms,
        bond_feat_dim,  # not used
        num_targets,
        act=swish,
        new_gnn: bool = True,
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = False,
        num_gaussians: int = 50,
        num_filters: int = 128,
        num_interactions: int = 6,
        cutoff: float = 10.0,
        hidden_channels: int = 128,
        tag_hidden_channels: int = 32,
        pg_hidden_channels: int = 32,
        phys_hidden_channels: int = 0,
        phys_embeds: bool = False,
        graph_rewiring=False,
        energy_head=False,
    ):
        super(SfariNet, self).__init__()
        self.cutoff = cutoff
        self.act = act
        self.use_pbc = use_pbc
        self.hidden_channels = hidden_channels
        self.tag_hidden_channels = tag_hidden_channels
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_hidden_channels = phys_hidden_channels
        self.phys_embeds = phys_embeds
        self.regress_forces = regress_forces
        self.graph_rewiring = graph_rewiring
        self.energy_head = energy_head
        self.use_positional_embeds = graph_rewiring in {
            "one-supernode-per-graph",
            "one-supernode-per-atom-type",
            "one-supernode-per-atom-type-dist",
        }
        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Embedding block
        self.embed_block = EmbeddingBlock(
            num_gaussians,
            hidden_channels,
            tag_hidden_channels,
            pg_hidden_channels,
            phys_hidden_channels,
            phys_embeds,
            graph_rewiring,
            act,
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels,
                    act,
                )
                for _ in range(num_interactions)
            ]
        )

        # Output block
        self.output_block = OutputBlock(energy_head, hidden_channels, act)
        
        if self.energy_head == 'weighted-av-initial-embeds':
            self.w_lin = Linear(hidden_channels, 1)

    def forward(self, data):
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
        rel_pos_normalized = rel_pos / edge_weight.view(-1, 1)
        rel_pos_normalized = (rel_pos_normalized + 1) / 2.0

        pooling_loss = None  # deal with pooling loss

        # Embedding block
        h, e = self.embed_block(z, rel_pos, edge_attr, data.tags)

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

        if self.regress_forces:
            force = self.decoder(h)
            return energy, force

        return energy, pooling_loss
