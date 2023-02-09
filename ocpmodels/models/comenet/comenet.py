from torch_cluster import radius_graph
from torch_geometric.nn import GraphConv, GraphNorm
from torch_geometric.nn import inits

from torch_scatter import scatter, scatter_min

try:
    from ocpmodels.common.registry import registry
    from ocpmodels.common.utils import conditional_grad, get_pbc_distances, radius_graph_pbc
    from .utils import angle_emb, torsion_emb
except:
    print("No OCP framework detected")

from torch.nn import Embedding

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import math
from math import sqrt

try:
    import sympy as sym
except ImportError:
    sym = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def swish(x):
    return x * torch.sigmoid(x)

class Linear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 weight_initializer='glorot',
                 bias_initializer='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        assert in_channels > 0
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels > 0:
            if self.weight_initializer == 'glorot':
                inits.glorot(self.weight)
            elif self.weight_initializer == 'glorot_orthogonal':
                inits.glorot_orthogonal(self.weight, scale=2.0)
            elif self.weight_initializer == 'uniform':
                bound = 1.0 / math.sqrt(self.weight.size(-1))
                torch.nn.init.uniform_(self.weight.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            elif self.weight_initializer == 'zeros':
                inits.zeros(self.weight)
            elif self.weight_initializer is None:
                inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(
                    f"Linear layer weight initializer "
                    f"'{self.weight_initializer}' is not supported")

        if self.in_channels > 0 and self.bias is not None:
            if self.bias_initializer == 'zeros':
                inits.zeros(self.bias)
            elif self.bias_initializer is None:
                inits.uniform(self.in_channels, self.bias)
            else:
                raise RuntimeError(
                    f"Linear layer bias initializer "
                    f"'{self.bias_initializer}' is not supported")

    def forward(self, x, _):
        """"""
        return F.linear(x, self.weight, self.bias)


class HeteroLinear(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 num_tags: int, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, **kwargs)
            for _ in range(num_tags)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, node_tag: Tensor) -> Tensor:
        """"""
        out = x.new_empty(x.size(0), self.out_channels)
        for i, lin in enumerate(self.lins):
            mask = node_tag == i
            out[mask] = lin(x[mask], None)
        return out


class TwoLayerLinear(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False,
            hetero=False
    ):
        super(TwoLayerLinear, self).__init__()
        if hetero:
            self.lin1 = HeteroLinear(in_channels, middle_channels, num_tags=3, bias=bias)
            self.lin2 = HeteroLinear(middle_channels, out_channels, num_tags=3, bias=bias)
        else:
            self.lin1 = Linear(in_channels, middle_channels, bias=bias)
            self.lin2 = Linear(middle_channels, out_channels, bias=bias)

        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, tags):
        x = self.lin1(x, tags)
        if self.act:
            x = swish(x)
        x = self.lin2(x, tags)
        if self.act:
            x = swish(x)
        return x


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x = self.act(self.emb(x))
        return x


class EdgeGraphConv(GraphConv):

    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight * x_j


class SimpleInteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            num_radial,
            num_spherical,
            num_layers,
            output_channels,
            act=swish,
            hetero=False,
            inits='glorot',
    ):
        super(SimpleInteractionBlock, self).__init__()
        self.act = act

        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lin_cat = Linear(2 * hidden_channels, hidden_channels)

        self.norm = GraphNorm(hidden_channels)

        # Transformations of Bessel and spherical basis representations.
        self.lin_feature1 = TwoLayerLinear(num_radial * num_spherical ** 2, hidden_channels, hidden_channels, hetero=hetero)
        self.lin_feature2 = TwoLayerLinear(num_radial * num_spherical, hidden_channels, hidden_channels, hetero=hetero)

        # Dense transformations of input messages.
        if hetero:
            self.lin = HeteroLinear(hidden_channels, hidden_channels, num_tags=3)
            self.lins = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.lins.append(HeteroLinear(hidden_channels, hidden_channels, num_tags=3))
            self.final = HeteroLinear(hidden_channels, output_channels, num_tags=3, weight_initializer=inits)
        else:
            self.lin = Linear(hidden_channels, hidden_channels)
            self.lins = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.lins.append(Linear(hidden_channels, hidden_channels))
            self.final = Linear(hidden_channels, output_channels, weight_initializer=inits)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.norm.reset_parameters()

        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.lin_cat.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(self, x, feature1, feature2, edge_index, batch, tags):
        x = self.act(self.lin(x, tags))

        feature1 = self.lin_feature1(feature1, tags)
        h1 = self.conv1(x, edge_index, feature1)
        h1 = self.lin1(h1, tags)
        h1 = self.act(h1)

        feature2 = self.lin_feature2(feature2, tags)
        h2 = self.conv2(x, edge_index, feature2)
        h2 = self.lin2(h2, tags)
        h2 = self.act(h2)

        h = self.lin_cat(torch.cat([h1, h2], 1), tags)

        h = h + x
        for lin in self.lins:
            h = self.act(lin(h, tags)) + h
        h = self.norm(h, batch)
        h = self.final(h, tags)
        return h


@registry.register_model("comenet")
class ComENet(nn.Module):
    def __init__(
            self,
            num_atoms,
            bond_feat_dim,  # not used
            num_targets=1,
            otf_graph=False,
            use_pbc=True,
            regress_forces=False,
            hidden_channels=128,
            num_blocks=4,
            num_radial=32,
            num_spherical=7,
            cutoff=6.0,
            num_output_layers=3,
            hetero=False,
    ):
        super(ComENet, self).__init__()
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.num_blocks = num_blocks

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        act = swish
        self.act = act

        self.feature1 = torsion_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature2 = angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        self.emb = EmbeddingBlock(hidden_channels, act)

        self.interaction_blocks = torch.nn.ModuleList(
            [
                SimpleInteractionBlock(
                    hidden_channels,
                    num_radial,
                    num_spherical,
                    num_output_layers,
                    hidden_channels,
                    act,
                    hetero=hetero,
                )
                for _ in range(num_blocks)
            ]
        )
        self.lins = torch.nn.ModuleList()
        if hetero:
            for _ in range(num_output_layers):
                self.lins.append(HeteroLinear(hidden_channels, hidden_channels, num_tags=3))
            self.lin_out = HeteroLinear(hidden_channels, num_targets, num_tags=3, weight_initializer='zeros')
        else:
            for _ in range(num_output_layers):
                self.lins.append(Linear(hidden_channels, hidden_channels))
            self.lin_out = Linear(hidden_channels, num_targets, weight_initializer='zeros')
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        tags = data.tags
        batch = data.batch
        z = data.atomic_numbers.long()
        num_nodes = data.atomic_numbers.size(0)

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50
            )

            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            out = get_pbc_distances(
                data.pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
                return_distance_vec=True
            )

            edge_index = out["edge_index"]
            j, i = edge_index
            dist = out["distances"]
            vecs = out["distance_vec"]
        else:
            pos = data.pos
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=1000)
            j, i = edge_index
            vecs = pos[j] - pos[i]
            dist = vecs.norm(dim=-1)

            mask = data.real_mask

        # Embedding block.
        x = self.emb(z)

        # Calculate distances.
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
        argmin0[argmin0 >= len(i)] = 0
        n0 = j[argmin0]
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]
        # --------------------------------------------------------

        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        # i[argmin] = range(0, num_nodes)
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]

        # ----------------------------------------------------------

        # n0, n1 for i
        n0 = n0[i]
        n1 = n1[i]

        # n0, n1 for j
        n0_j = n0_j[j]
        n1_j = n1_j[j]

        # tau: (iref, i, j, jref)
        # when compute tau, do not use n0, n0_j as ref for i and j,
        # because if n0 = j, or n0_j = i, the computed tau is zero
        # so if n0 = j, we choose iref = n1
        # if n0_j = i, we choose jref = n1_j
        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]

        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # Calculate angles.
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi[phi < 0] = phi[phi < 0] + math.pi

        # Calculate right torsions.
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau[tau < 0] = tau[tau < 0] + math.pi

        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)

        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature1, feature2, edge_index, batch, tags)

        for lin in self.lins:
            x = self.act(lin(x, tags))
        x = self.lin_out(x, tags)

        if self.use_pbc:
            energy = scatter(x, batch, dim=0)
        else:
            energy = scatter(x[mask], batch[mask], dim=0)
        return energy

    def forward(self, data):
        return self._forward(data)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
