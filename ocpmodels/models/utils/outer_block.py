from typing import Callable, Optional

import torch
from e3nn import o3
from torch import nn
from torch_scatter import scatter


class Rank2Block(nn.Module):
    r"""Prediction of rank 2 tensor
    Applies outer product between edges and compute node-wise or edge-wise MLP
    Parameters
    ----------
    edge_level : bool
        apply MLP to edges' outer product
    mixing_coordinates : bool
        apply MLP to the channels or to the 9 components. If True we loose equivariance
    emb_size : int
        size of edge embedding used to compute outer products
    num_layers : int
        number of layers of the MLP
    --------
    """

    def __init__(
        self,
        edge_level,
        mixing_coordinates,
        emb_size,
        extensive=False,
        num_layers=2,
    ):
        super().__init__()

        self.edge_level = edge_level
        self.mixing_coordinates = mixing_coordinates
        self.emb_size = emb_size
        self.extensive = extensive
        self.scalar_nonlinearity = nn.SiLU()
        self.stress_MLP = nn.ModuleList()
        for i in range(num_layers):
            if self.mixing_coordinates:
                if i < num_layers - 1:
                    self.stress_MLP.append(
                        nn.Linear(emb_size * 9, emb_size * 9)
                    )
                    self.stress_MLP.append(self.scalar_nonlinearity)
                else:
                    self.stress_MLP.append(nn.Linear(emb_size * 9, 9))
            else:
                if i < num_layers - 1:
                    self.stress_MLP.append(nn.Linear(emb_size, emb_size))
                    self.stress_MLP.append(self.scalar_nonlinearity)
                else:
                    self.stress_MLP.append(nn.Linear(emb_size, 1))

    def forward(self, edge_distance_vec, x_edge, edge_index, data):
        """evaluate
        Parameters
        ----------
        edge_distance_vec : `torch.Tensor`
            tensor of shape ``(..., 3)``
        x_edge : `torch.Tensor`
            tensor of shape ``(..., emb_size)``
        edge_index : `torch.Tensor`
            tensor of shape ``(...)``
        data : ``LMDBDataset sample``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., 9)``
        """

        outer_product_edge = torch.bmm(
            edge_distance_vec.unsqueeze(2), edge_distance_vec.unsqueeze(1)
        )
        edge_outer = (
            x_edge[:, :, None] * outer_product_edge.view(-1, 9)[:, None, :]
        )
        nEdges = edge_outer.shape[0]
        if self.edge_level:
            # edge_outer: (nEdges, emb_size_edge, 9)
            ## operates at edge level before mixing / MLP => mixing / MLP happens at NOde level
            if self.mixing_coordinates:
                edge_outer = edge_outer.view(
                    nEdges, -1
                )  # (nEdges, 9 * emb_size_edge)
                for module in self.stress_MLP:
                    edge_outer = module(edge_outer)  # (nEdges,  9)

            else:
                # self.stress_MLP MLP to be defined
                edge_outer = edge_outer.transpose(
                    1, 2
                )  # (nEdges, 9, emb_size_edge)
                for module in self.stress_MLP:
                    edge_outer = module(edge_outer)  # (nEdges, 9, 1)
                edge_outer = edge_outer.reshape(-1, 9)  # (nEdges, 9)

            node_outer = scatter(edge_outer, edge_index, dim=0, reduce="mean")

        else:
            # edge_outer: (nEdges, emb_size_edge, 9)
            ## operates at edge level before mixing / MLP => mixing / MLP happens at NOde level
            node_outer = scatter(edge_outer, edge_index, dim=0, reduce="mean")
            if self.mixing_coordinates:
                node_outer = node_outer.view(
                    len(data.fixed), -1
                )  # (natoms, 9 * emb_size_edge)
                for module in self.stress_MLP:
                    node_outer = module(
                        node_outer
                    )  # (nAtoms,  num_targets) # stress_MLP : emb -> num_targets (9)
            else:
                node_outer = node_outer.transpose(
                    1, 2
                )  # (natoms, 9, emb_size_edge)
                for module in self.stress_MLP:
                    node_outer = module(node_outer)  # (natoms, 9, 1)
                node_outer = node_outer.reshape(-1, 9)  # (natoms, 9)

        # node_outer: nAtoms, 9 => average across all atoms at the molecular level
        if self.extensive:
            stress = scatter(node_outer, data.batch, dim=0, reduce="sum")
        else:
            stress = scatter(node_outer, data.batch, dim=0, reduce="mean")
        return stress


class Rank2DecompositionBlock(nn.Module):
    r"""Prediction of rank 2 tensor
    Decompose rank 2 tensor with irreps
    since it is symmetric we need just irrep degree 0 and 2
    Parameters
    ----------
    emb_size : int
        size of edge embedding used to compute outer products
    num_layers : int
        number of layers of the MLP
    --------
    """

    def __init__(
        self,
        emb_size,
        extensive=False,
        num_layers=2,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.extensive = extensive
        self.scalar_nonlinearity = nn.SiLU()
        self.scalar_MLP = nn.ModuleList()
        self.irrep2_MLP = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.scalar_MLP.append(nn.Linear(emb_size, emb_size))
                self.irrep2_MLP.append(nn.Linear(emb_size, emb_size))
                self.scalar_MLP.append(self.scalar_nonlinearity)
                self.irrep2_MLP.append(self.scalar_nonlinearity)
            else:
                self.scalar_MLP.append(nn.Linear(emb_size, 1))
                self.irrep2_MLP.append(nn.Linear(emb_size, 1))

        # Change of basis obtained by stacking the C-G coefficients in the right way
        self.change_mat = torch.transpose(
            torch.tensor(
                [
                    [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                    [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                    [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                    [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                    [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                    [
                        -(6 ** (-0.5)),
                        0,
                        0,
                        0,
                        2 * 6 ** (-0.5),
                        0,
                        0,
                        0,
                        -(6 ** (-0.5)),
                    ],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                    [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
                ]
            ).detach(),
            0,
            1,
        )

    def forward(self, x_pointwise, sphere_points, data):
        """evaluate
        Parameters
        ----------
        x_pointwise : `torch.Tensor`
            tensor of shape ``(..., num_sphere_samples, emb_size)``
        sphere_points : `torch.Tensor`
            tensor of shape ``(num_sphere_samples, 3)``
        data : ``LMDBDataset sample``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., 9)``
        """
        # compute number of sphere samples
        num_sphere_samples = x_pointwise.shape[1]

        # Calculate spherical harmonics of degree 2 of the points sampled
        sphere_irrep2 = o3.spherical_harmonics(2, sphere_points, True).detach()

        # Irrep 0 prediction
        for i, module in enumerate(self.scalar_MLP):
            if i == 0:
                node_scalar = module(x_pointwise)
            else:
                node_scalar = module(node_scalar)
        node_scalar = node_scalar.view(-1, num_sphere_samples, 1)
        node_scalar = torch.sum(node_scalar, dim=1) / num_sphere_samples
        if self.extensive:
            scalar = scatter(
                node_scalar.view(-1), data.batch, dim=0, reduce="sum"
            )
        else:
            scalar = scatter(
                node_scalar.view(-1), data.batch, dim=0, reduce="mean"
            )

        # Irrep 2 prediction
        for i, module in enumerate(self.irrep2_MLP):
            if i == 0:
                node_irrep2 = module(x_pointwise)
            else:
                node_irrep2 = module(node_irrep2)
        node_irrep2 = node_irrep2 * sphere_irrep2.view(
            1, num_sphere_samples, 5
        )
        node_irrep2 = torch.sum(node_irrep2, dim=1) / num_sphere_samples
        if self.extensive:
            irrep2 = scatter(
                node_irrep2.view(-1, 5), data.batch, dim=0, reduce="sum"
            )
        else:
            irrep2 = scatter(
                node_irrep2.view(-1, 5), data.batch, dim=0, reduce="mean"
            )

        # Change of basis to compute a rank 2 symmetric tensor
        vector = torch.zeros(
            (len(data.natoms), 3), device=x_pointwise.device
        ).detach()
        flatten_irreps = torch.cat(
            [scalar.reshape(-1, 1), vector, irrep2], dim=1
        )
        # stress = torch.einsum("ab, cb->ca", self.change_mat.to(flatten_irreps.device), flatten_irreps)

        return scalar.reshape(-1), irrep2


class Rank2DecompositionEdgeBlock(nn.Module):
    r"""Prediction of rank 2 tensor
    Decompose rank 2 tensor with irreps
    since it is symmetric we need just irrep degree 0 and 2
    Parameters
    ----------
    emb_size : int
        size of edge embedding used to compute outer products
    num_layers : int
        number of layers of the MLP
    --------
    """

    def __init__(
        self,
        emb_size,
        edge_level,
        extensive=False,
        num_layers=2,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.edge_level = edge_level
        self.extensive = extensive
        self.scalar_nonlinearity = nn.SiLU()
        self.scalar_MLP = nn.ModuleList()
        self.irrep2_MLP = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.scalar_MLP.append(nn.Linear(emb_size, emb_size))
                self.irrep2_MLP.append(nn.Linear(emb_size, emb_size))
                self.scalar_MLP.append(self.scalar_nonlinearity)
                self.irrep2_MLP.append(self.scalar_nonlinearity)
            else:
                self.scalar_MLP.append(nn.Linear(emb_size, 1))
                self.irrep2_MLP.append(nn.Linear(emb_size, 1))

        # Change of basis obtained by stacking the C-G coefficients in the right way
        self.change_mat = torch.transpose(
            torch.tensor(
                [
                    [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                    [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                    [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                    [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                    [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                    [
                        -(6 ** (-0.5)),
                        0,
                        0,
                        0,
                        2 * 6 ** (-0.5),
                        0,
                        0,
                        0,
                        -(6 ** (-0.5)),
                    ],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                    [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
                ]
            ).detach(),
            0,
            1,
        )

    def forward(self, x_edge, egde_vec, edge_index, data):
        """evaluate
        Parameters
        ----------
        x_edge : `torch.Tensor`
            tensor of shape ``(nEdges, emb_size)``
        egde_vec : `torch.Tensor`
            tensor of shape ``(nEdges, 3)``
        data : ``LMDBDataset sample``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., 9)``
        """
        # Calculate spherical harmonics of degree 2 of the points sampled
        sphere_irrep2 = o3.spherical_harmonics(
            2, egde_vec, True
        ).detach()  # (nEdges, 5)

        if self.edge_level:
            # Irrep 0 prediction
            for i, module in enumerate(self.scalar_MLP):
                if i == 0:
                    edge_scalar = module(x_edge)
                else:
                    edge_scalar = module(edge_scalar)

            # Irrep 2 prediction
            edge_irrep2 = (
                sphere_irrep2[:, :, None] * x_edge[:, None, :]
            )  # (nEdges, 5, emb_size)
            for i, module in enumerate(self.irrep2_MLP):
                if i == 0:
                    edge_irrep2 = module(edge_irrep2)
                else:
                    edge_irrep2 = module(edge_irrep2)

            node_scalar = scatter(
                edge_scalar, edge_index, dim=0, reduce="mean"
            )
            node_irrep2 = scatter(
                edge_irrep2, edge_index, dim=0, reduce="mean"
            )
        else:
            edge_irrep2 = (
                sphere_irrep2[:, :, None] * x_edge[:, None, :]
            )  # (nAtoms, 5, emb_size)

            node_scalar = scatter(x_edge, edge_index, dim=0, reduce="mean")
            node_irrep2 = scatter(
                edge_irrep2, edge_index, dim=0, reduce="mean"
            )

            # Irrep 0 prediction
            for i, module in enumerate(self.scalar_MLP):
                if i == 0:
                    node_scalar = module(node_scalar)
                else:
                    node_scalar = module(node_scalar)

            # Irrep 2 prediction
            for i, module in enumerate(self.irrep2_MLP):
                if i == 0:
                    node_irrep2 = module(node_irrep2)
                else:
                    node_irrep2 = module(node_irrep2)

        if self.extensive:
            scalar = scatter(
                node_scalar.view(-1), data.batch, dim=0, reduce="sum"
            )
            irrep2 = scatter(
                node_irrep2.view(-1, 5), data.batch, dim=0, reduce="sum"
            )
        else:
            irrep2 = scatter(
                node_irrep2.view(-1, 5), data.batch, dim=0, reduce="mean"
            )
            scalar = scatter(
                node_scalar.view(-1), data.batch, dim=0, reduce="mean"
            )

        # Change of basis to compute a rank 2 symmetric tensor

        vector = torch.zeros(
            (len(data.natoms), 3), device=scalar.device
        ).detach()
        flatten_irreps = torch.cat(
            [scalar.reshape(-1, 1), vector, irrep2], dim=1
        )
        stress = torch.einsum(
            "ab, cb->ca",
            self.change_mat.to(flatten_irreps.device),
            flatten_irreps,
        )

        return scalar.reshape(-1), irrep2
