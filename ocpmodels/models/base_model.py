"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from ocpmodels.common.utils import (
    compute_neighbors,
    get_pbc_distances,
    radius_graph_pbc,
)


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def set_deup_inference(self, deup_inference):
        self.deup_inference = deup_inference

        def _sdi(module):
            if isinstance(module, nn.Module):
                module.deup_inference = deup_inference

        self.apply(_sdi)

    def set_dropouts(self, dropout):
        def _sds(module):
            if hasattr(module, "dropout_lin"):
                module.dropout_lin = dropout
            if hasattr(module, "dropout_edge"):
                module.dropout_edge = dropout

        self.apply(_sds)

    @staticmethod
    def freeze_layer(layer):
        for param in layer.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                assert isinstance(child, nn.Module)
                assert callable(child.reset_parameters)
                child.reset_parameters()
            else:
                if hasattr(child, "weight"):
                    nn.init.xavier_uniform_(child.weight)
                if hasattr(child, "bias"):
                    child.bias.data.fill_(0)

    def energy_forward(self, data):
        """Forward pass for energy prediction."""
        raise NotImplementedError

    def forces_forward(self, preds):
        """Forward pass for force prediction."""
        raise NotImplementedError

    def forward(self, data, mode="train", regress_forces=None, q=None):
        grad_forces = forces = None

        # Fine tune on gradients
        if regress_forces:
            self.regress_forces = regress_forces

        # energy gradient w.r.t. positions will be computed
        if mode == "train" or self.regress_forces == "from_energy":
            data.pos.requires_grad_(True)

        # predict energy
        preds = self.energy_forward(data, q=q)

        if self.regress_forces:
            if self.regress_forces in {"direct", "direct_with_gradient_target"}:
                # predict forces
                forces = self.forces_forward(preds)

            if mode == "train" or self.regress_forces == "from_energy":
                grad_forces = self.forces_as_energy_grad(data.pos, preds["energy"])

            if self.regress_forces == "from_energy":
                # predicted forces are the energy gradient
                preds["forces"] = grad_forces
            elif self.regress_forces in {"direct", "direct_with_gradient_target"}:
                # predicted forces are the model's direct forces
                preds["forces"] = forces
                if mode == "train":
                    # Store the energy gradient as target for "direct_with_gradient_target"
                    # Use it as a metric only in "direct" mode.
                    preds["forces_grad_target"] = grad_forces.detach()
            else:
                raise ValueError(
                    f"Unknown forces regression mode {self.regress_forces}"
                )

        return preds

    def forces_as_energy_grad(self, pos, energy):
        """Computes forces from energy gradient

        Args:
            pos (tensor): atom positions
            energy (tensor): predicted energy

        Returns:
            forces (tensor): gradient of energy w.r.t. atom positions
        """

        return -1 * (
            torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
    ):
        # from https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/base.py
        # after importing gemnet_oc
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present "
                    + "in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data, cutoff, max_neighbors
                )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )
