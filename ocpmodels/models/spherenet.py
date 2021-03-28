"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

This code borrows heavily from the SphereNet implementation as part of DIG:
Dive into Graphs: https://github.com/divelab/DIG. License: GPL-3.0.

"""

import os
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn import radius_graph
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.models.utils.spherenet_utils.features import (
    angle_emb,
    dist_emb,
    torsion_emb,
    xyztodat,
)

try:
    import sympy as sym
except ImportError:
    sym = None


class emb(torch.nn.Module):
    def __init__(
        self, num_spherical, num_radial, cutoff, envelope_exponent, torsional
    ):
        super(emb, self).__init__()
        self.torsional = torsional
        self.dist_emb = dist_emb(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(
            num_spherical, num_radial, cutoff, envelope_exponent
        )
        if torsional:
            self.torsion_emb = torsion_emb(
                num_spherical, num_radial, cutoff, envelope_exponent
            )
        self.reset_parameters()

    def reset_parameters(self):
        self.dist_emb.reset_parameters()

    def forward(self, dist, angle, torsion, idx_kj):
        dist_emb = self.dist_emb(dist)
        angle_emb = self.angle_emb(dist, angle, idx_kj)
        if self.torsional:
            torsion_emb = self.torsion_emb(dist, angle, torsion, idx_kj)
            return dist_emb, angle_emb, torsion_emb
        else:
            return dist_emb, angle_emb, None


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super(init, self).__init__()
        self.act = act
        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf_0 = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf, _, _ = emb
        x = self.emb(x.long())
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1

        return e1, e2


class update_e(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        torsional,
        act=swish,
    ):
        super(update_e, self).__init__()
        self.torsional = torsional
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_emb_size, bias=False
        )
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        if torsional:
            self.lin_t1 = nn.Linear(
                num_spherical * num_spherical * num_radial,
                basis_emb_size,
                bias=False,
            )
            self.lin_t2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_before_skip)
            ]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_after_skip)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        if self.torsional:
            glorot_orthogonal(self.lin_t1.weight, scale=2.0)
            glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1, _ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        if self.torsional:
            t = self.lin_t1(t)
            t = self.lin_t2(t)
            x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1

        return e1, e2


class update_v(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_output_layers,
        act,
        output_init,
    ):
        super(update_v, self).__init__()
        self.act = act
        self.output_init = output_init

        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_output_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_init == "zeros":
            self.lin.weight.data.fill_(0)
        if self.output_init == "GlorotOrthogonal":
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, e, i, num_nodes=None):
        _, e2 = e
        v = scatter(e2, i, dim=0)  # , dim_size=num_nodes
        v = self.lin_up(v)
        for lin in self.lins:
            v = self.act(lin(v))
        v = self.lin(v)
        return v


class update_u(torch.nn.Module):
    def __init__(self):
        super(update_u, self).__init__()

    def forward(self, u, v, batch):
        u += scatter(v, batch, dim=0)
        return u


@registry.register_model("spherenet")
class SphereNet(torch.nn.Module):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        num_layers=4,
        hidden_channels=128,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
        use_torsional=True,
        output_init="GlorotOrthogonal",
        regress_forces=True,
        use_pbc=True,
        otf_graph=False,
        cutoff=6.0,
    ):
        super(SphereNet, self).__init__()

        self.cutoff = cutoff
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.torsional = use_torsional

        self.init_e = init(num_radial, hidden_channels, act)
        self.init_v = update_v(
            hidden_channels,
            out_emb_channels,
            num_targets,
            num_output_layers,
            act,
            output_init,
        )
        self.init_u = update_u()
        self.emb = emb(
            num_spherical,
            num_radial,
            self.cutoff,
            envelope_exponent,
            self.torsional,
        )

        self.update_vs = torch.nn.ModuleList(
            [
                update_v(
                    hidden_channels,
                    out_emb_channels,
                    num_targets,
                    num_output_layers,
                    act,
                    output_init,
                )
                for _ in range(num_layers)
            ]
        )

        self.update_es = torch.nn.ModuleList(
            [
                update_e(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    self.torsional,
                    act,
                )
                for _ in range(num_layers)
            ]
        )

        self.update_us = torch.nn.ModuleList(
            [update_u() for _ in range(num_layers)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.init_e.reset_parameters()
        self.init_v.reset_parameters()
        self.emb.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z, pos, batch = data.atomic_numbers, data.pos, data.batch
        num_nodes = z.size(0)

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50, data.pos.device
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
                return_offsets=True,
            )
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            out = {"edge_index": edge_index}

        dist, angle, torsion, i, j, idx_kj, idx_ji = xyztodat(
            pos, out, num_nodes, use_pbc=self.use_pbc, torsional=self.torsional
        )

        emb = self.emb(dist, angle, torsion, idx_kj)

        # Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i, num_nodes=pos.size(0))
        u = self.init_u(
            torch.zeros_like(scatter(v, batch, dim=0)), v, batch
        )  # scatter(v, batch, dim=0)

        for update_e, update_v, update_u in zip(
            self.update_es, self.update_vs, self.update_us
        ):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i)
            u = update_u(u, v, batch)  # u += scatter(v, batch, dim=0)

        return u

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

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
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
