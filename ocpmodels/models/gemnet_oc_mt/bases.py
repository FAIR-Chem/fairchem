from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from ocpmodels.common.typing import TypedModuleList

from .config import BasesConfig
from .layers.base_layers import Dense
from .layers.efficient import BasisEmbedding
from .layers.embedding_block import EdgeEmbedding
from .layers.radial_basis_dynamic_cutoff import GaussianBasis, RadialBasis
from .layers.spherical_basis_dynamic_cutoff import (
    CircularBasisLayer,
    SphericalBasisLayer,
)
from .utils import get_angle, inner_product_clamped

TripletIn = TypedDict(
    "TripletIn",
    {"adj_edges": SparseTensor, "in": torch.Tensor, "out": torch.Tensor},
)
TripletOut = TypedDict("TripletOut", {"in": torch.Tensor, "out": torch.Tensor})


class QuadIdx(TypedDict):
    triplet_in: TripletIn
    triplet_out: TripletOut
    out: torch.Tensor
    trip_out_to_quad: torch.Tensor
    trip_in_to_quad: torch.Tensor
    out_agg: torch.Tensor


class GraphBases(TypedDict):
    rad: torch.Tensor
    cir: list[torch.Tensor]
    # cir: tuple[torch.Tensor, torch.Tensor]


class GraphBasesQInt(TypedDict):
    rad: torch.Tensor
    cir: torch.Tensor
    sph: list[torch.Tensor]
    # sph: tuple[torch.Tensor, torch.Tensor]


@dataclass
class BasesOutput:
    m: torch.Tensor  # (e_main, emb_size_edge)
    atom_update: torch.Tensor  # (e_main, emb_size_rbf)
    output: torch.Tensor  # (e_main, emb_size_rbf)
    qint: GraphBasesQInt
    e2e: GraphBases
    a2e: GraphBases
    e2a: GraphBases
    a2a_rad: torch.Tensor | None


class Bases(nn.Module):
    def __init__(self, config: BasesConfig):
        super().__init__()

        self.config = config

        self.init_basis_functions()
        self.init_shared_basis_layers()

        self.edge_emb = EdgeEmbedding(
            self.config.emb_size_atom,
            self.config.num_radial,
            self.config.emb_size_edge,
            activation=self.config.activation,
        )

        if not self.config.unique_per_layer:
            self._set_shared_params()

    def _set_shared_params(self):
        # Set shared parameters for better gradients
        self.shared_parameters: list[tuple[nn.Parameter, float | int]] = []
        self.shared_parameters += [
            (self.mlp_rbf_tint.linear.weight, self.config.num_blocks),
            (self.mlp_cbf_tint.weight, self.config.num_blocks),
            (self.mlp_rbf_h.linear.weight, self.config.num_blocks),
            (self.mlp_rbf_out.linear.weight, self.config.num_blocks + 1),
        ]
        if self.config.quad_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_qint.linear.weight, self.config.num_blocks),
                (self.mlp_cbf_qint.weight, self.config.num_blocks),
                (self.mlp_sbf_qint.weight, self.config.num_blocks),
            ]
        if self.config.atom_edge_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_aeint.linear.weight, self.config.num_blocks),
                (self.mlp_cbf_aeint.weight, self.config.num_blocks),
            ]
        if self.config.edge_atom_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_eaint.linear.weight, self.config.num_blocks),
                (self.mlp_cbf_eaint.weight, self.config.num_blocks),
            ]
        if self.config.atom_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_aint.weight, self.config.num_blocks),
            ]

        self._add_rbf_shared_params(self.radial_basis)
        if self.config.quad_interaction:
            self._add_rbf_shared_params(self.cbf_basis_qint.radial_basis)
            self._add_rbf_shared_params(self.sbf_basis_qint.radial_basis)

        if self.config.atom_edge_interaction:
            self._add_rbf_shared_params(self.cbf_basis_aeint.radial_basis)
        if self.config.edge_atom_interaction:
            self._add_rbf_shared_params(self.cbf_basis_eaint.radial_basis)
        if self.config.atom_interaction:
            self._add_rbf_shared_params(self.radial_basis_aint)
        self._add_rbf_shared_params(self.cbf_basis_tint.radial_basis)

    def _add_rbf_shared_params(self, radial: RadialBasis):
        match radial.rbf:
            case GaussianBasis(trainable=True) as gbf:
                for param in gbf.parameters():
                    self._add_shared_param(param, self.config.num_blocks)
            case _:
                pass

    def _add_shared_param(self, param: nn.Parameter, factor: int | float):
        if (
            shared_param_idx := next(
                (
                    i
                    for i, p in enumerate(self.shared_parameters)
                    if p[0] is param
                ),
                None,
            )
        ) is not None:
            self.shared_parameters[shared_param_idx] = (
                param,
                self.shared_parameters[shared_param_idx][1] + factor,
            )
        else:
            self.shared_parameters += [(param, factor)]

    def init_basis_functions(self):
        self.radial_basis = RadialBasis(
            num_radial=self.config.num_radial,
            graph_type="main",
            rbf=self.config.rbf,
            envelope=self.config.envelope,
            scale_basis=self.config.scale_basis,
            absolute_cutoff=self.config.absolute_rbf_cutoff,
        )
        radial_basis_spherical = RadialBasis(
            num_radial=self.config.num_radial,
            graph_type="main",
            rbf=self.config.rbf_spherical,
            envelope=self.config.envelope,
            scale_basis=self.config.scale_basis,
            absolute_cutoff=self.config.absolute_rbf_cutoff,
        )
        if self.config.quad_interaction:
            radial_basis_spherical_qint = RadialBasis(
                num_radial=self.config.num_radial,
                graph_type="qint",
                rbf=self.config.rbf_spherical,
                envelope=self.config.envelope,
                scale_basis=self.config.scale_basis,
                absolute_cutoff=self.config.absolute_rbf_cutoff,
            )
            self.cbf_basis_qint = CircularBasisLayer(
                self.config.num_spherical,
                radial_basis=radial_basis_spherical_qint,
                cbf=self.config.cbf,
                scale_basis=self.config.scale_basis,
            )

            self.sbf_basis_qint = SphericalBasisLayer(
                self.config.num_spherical,
                radial_basis=radial_basis_spherical,
                sbf=self.config.sbf,
                scale_basis=self.config.scale_basis,
            )
        if self.config.atom_edge_interaction:
            self.radial_basis_aeaint = RadialBasis(
                num_radial=self.config.num_radial,
                graph_type="a2ee2a",
                rbf=self.config.rbf,
                envelope=self.config.envelope,
                scale_basis=self.config.scale_basis,
                absolute_cutoff=self.config.absolute_rbf_cutoff,
            )
            self.cbf_basis_aeint = CircularBasisLayer(
                self.config.num_spherical,
                radial_basis=radial_basis_spherical,
                cbf=self.config.cbf,
                scale_basis=self.config.scale_basis,
            )
        if self.config.edge_atom_interaction:
            self.radial_basis_aeaint = RadialBasis(
                num_radial=self.config.num_radial,
                graph_type="a2ee2a",
                rbf=self.config.rbf,
                envelope=self.config.envelope,
                scale_basis=self.config.scale_basis,
                absolute_cutoff=self.config.absolute_rbf_cutoff,
            )
            radial_basis_spherical_aeaint = RadialBasis(
                num_radial=self.config.num_radial,
                graph_type="a2ee2a",
                rbf=self.config.rbf_spherical,
                envelope=self.config.envelope,
                scale_basis=self.config.scale_basis,
                absolute_cutoff=self.config.absolute_rbf_cutoff,
            )
            self.cbf_basis_eaint = CircularBasisLayer(
                self.config.num_spherical,
                radial_basis=radial_basis_spherical_aeaint,
                cbf=self.config.cbf,
                scale_basis=self.config.scale_basis,
            )
        if self.config.atom_interaction:
            self.radial_basis_aint = RadialBasis(
                num_radial=self.config.num_radial,
                graph_type="a2a",
                rbf=self.config.rbf,
                envelope=self.config.envelope,
                scale_basis=self.config.scale_basis,
                absolute_cutoff=self.config.absolute_rbf_cutoff,
            )

        self.cbf_basis_tint = CircularBasisLayer(
            self.config.num_spherical,
            radial_basis=radial_basis_spherical,
            cbf=self.config.cbf,
            scale_basis=self.config.scale_basis,
        )

    def init_shared_basis_layers(self):
        # Share basis down projections across all interaction blocks
        if self.config.quad_interaction:
            self.mlp_rbf_qint = Dense(
                self.config.num_radial,
                self.config.emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.mlp_cbf_qint = BasisEmbedding(
                self.config.num_radial,
                self.config.emb_size_cbf,
                self.config.num_spherical,
            )
            self.mlp_sbf_qint = BasisEmbedding(
                self.config.num_radial,
                self.config.emb_size_sbf,
                self.config.num_spherical**2,
            )

        if self.config.atom_edge_interaction:
            self.mlp_rbf_aeint = Dense(
                self.config.num_radial,
                self.config.emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.mlp_cbf_aeint = BasisEmbedding(
                self.config.num_radial,
                self.config.emb_size_cbf,
                self.config.num_spherical,
            )
        if self.config.edge_atom_interaction:
            self.mlp_rbf_eaint = Dense(
                self.config.num_radial,
                self.config.emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.mlp_cbf_eaint = BasisEmbedding(
                self.config.num_radial,
                self.config.emb_size_cbf,
                self.config.num_spherical,
            )
        if self.config.atom_interaction:
            self.mlp_rbf_aint = BasisEmbedding(
                self.config.num_radial, self.config.emb_size_rbf
            )

        self.mlp_rbf_tint = Dense(
            self.config.num_radial,
            self.config.emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf_tint = BasisEmbedding(
            self.config.num_radial,
            self.config.emb_size_cbf,
            self.config.num_spherical,
        )

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            self.config.num_radial,
            self.config.emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            self.config.num_radial,
            self.config.emb_size_rbf,
            activation=None,
            bias=False,
        )

    def calculate_quad_angles(
        self,
        V_st: torch.Tensor,
        V_qint_st: torch.Tensor,
        quad_idx: dict,
    ):
        """Calculate angles for quadruplet-based message passing.

        Arguments
        ---------
        V_st: Tensor, shape = (nAtoms, 3)
            Normalized directions from s to t
        V_qint_st: Tensor, shape = (nAtoms, 3)
            Normalized directions from s to t for the quadruplet
            interaction graph
        quad_idx: dict of torch.Tensor
            Indices relevant for quadruplet interactions.

        Returns
        -------
        cosφ_cab: Tensor, shape = (num_triplets_inint,)
            Cosine of angle between atoms c -> a <- b.
        cosφ_abd: Tensor, shape = (num_triplets_qint,)
            Cosine of angle between atoms a -> b -> d.
        angle_cabd: Tensor, shape = (num_quadruplets,)
            Dihedral angle between atoms c <- a-b -> d.
        """
        # ---------------------------------- d -> b -> a ---------------------------------- #
        V_ba = V_qint_st[quad_idx["triplet_in"]["out"]]
        # (num_triplets_qint, 3)
        V_db = V_st[quad_idx["triplet_in"]["in"]]
        # (num_triplets_qint, 3)
        cosφ_abd = inner_product_clamped(V_ba, V_db)
        # (num_triplets_qint,)

        # Project for calculating dihedral angle
        # Cross product is the same as projection, just 90° rotated
        V_db_cross = torch.cross(V_db, V_ba, dim=-1)  # a - b -| d
        V_db_cross = V_db_cross[quad_idx["trip_in_to_quad"]]
        # (num_quadruplets,)

        # --------------------------------- c -> a <- b ---------------------------------- #
        V_ca = V_st[quad_idx["triplet_out"]["out"]]  # (num_triplets_in, 3)
        V_ba = V_qint_st[quad_idx["triplet_out"]["in"]]  # (num_triplets_in, 3)
        cosφ_cab = inner_product_clamped(V_ca, V_ba)  # (n4Triplets,)

        # Project for calculating dihedral angle
        # Cross product is the same as projection, just 90° rotated
        V_ca_cross = torch.cross(V_ca, V_ba, dim=-1)  # c |- a - b
        V_ca_cross = V_ca_cross[quad_idx["trip_out_to_quad"]]
        # (num_quadruplets,)

        # -------------------------------- c -> a - b <- d -------------------------------- #
        half_angle_cabd = get_angle(V_ca_cross, V_db_cross)
        # (num_quadruplets,)
        angle_cabd = half_angle_cabd
        # Ignore parity and just use the half angle.

        return cosφ_cab, cosφ_abd, angle_cabd

    def forward(
        self,
        data: Batch,
        *,
        h: torch.Tensor,
        main_graph: dict,
        a2a_graph: dict,
        a2ee2a_graph: dict,
        qint_graph: dict,
        trip_idx_e2e: dict,
        trip_idx_a2e: dict,
        trip_idx_e2a: dict,
        quad_idx: dict,
        num_atoms: int,
    ):
        """Calculate and transform basis functions."""
        basis_rad_main_raw = self.radial_basis(
            main_graph["distance"], data=data
        )

        # Calculate triplet angles
        cosφ_cab = inner_product_clamped(
            main_graph["vector"][trip_idx_e2e["out"]],
            main_graph["vector"][trip_idx_e2e["in"]],
        )
        basis_rad_cir_e2e_raw, basis_cir_e2e_raw = self.cbf_basis_tint(
            main_graph["distance"],
            cosφ_cab,
            data=data,
        )

        if self.config.quad_interaction:
            # Calculate quadruplet angles
            cosφ_cab_q, cosφ_abd, angle_cabd = self.calculate_quad_angles(
                main_graph["vector"],
                qint_graph["vector"],
                quad_idx,
            )

            basis_rad_cir_qint_raw, basis_cir_qint_raw = self.cbf_basis_qint(
                qint_graph["distance"],
                cosφ_abd,
                data=data,
            )
            basis_rad_sph_qint_raw, basis_sph_qint_raw = self.sbf_basis_qint(
                main_graph["distance"],
                cosφ_cab_q[quad_idx["trip_out_to_quad"]],
                angle_cabd,
                data=data,
            )
        if self.config.atom_edge_interaction:
            basis_rad_a2ee2a_raw = self.radial_basis_aeaint(
                a2ee2a_graph["distance"],
                data=data,
            )
            cosφ_cab_a2e = inner_product_clamped(
                main_graph["vector"][trip_idx_a2e["out"]],
                a2ee2a_graph["vector"][trip_idx_a2e["in"]],
            )
            basis_rad_cir_a2e_raw, basis_cir_a2e_raw = self.cbf_basis_aeint(
                main_graph["distance"],
                cosφ_cab_a2e,
                data=data,
            )
        if self.config.edge_atom_interaction:
            cosφ_cab_e2a = inner_product_clamped(
                a2ee2a_graph["vector"][trip_idx_e2a["out"]],
                main_graph["vector"][trip_idx_e2a["in"]],
            )
            basis_rad_cir_e2a_raw, basis_cir_e2a_raw = self.cbf_basis_eaint(
                a2ee2a_graph["distance"],
                cosφ_cab_e2a,
                data=data,
            )
        if self.config.atom_interaction:
            basis_rad_a2a_raw = self.radial_basis_aint(
                a2a_graph["distance"],
                data=data,
            )

        # Shared Down Projections
        bases_qint: GraphBasesQInt = {}
        if self.config.quad_interaction:
            bases_qint["rad"] = self.mlp_rbf_qint(basis_rad_main_raw)
            bases_qint["cir"] = self.mlp_cbf_qint(
                rad_basis=basis_rad_cir_qint_raw,
                sph_basis=basis_cir_qint_raw,
                idx_sph_outer=quad_idx["triplet_in"]["out"],
            )
            bases_qint["sph"] = list(
                self.mlp_sbf_qint(
                    rad_basis=basis_rad_sph_qint_raw,
                    sph_basis=basis_sph_qint_raw,
                    idx_sph_outer=quad_idx["out"],
                    idx_sph_inner=quad_idx["out_agg"],
                )
            )

        bases_a2e: GraphBases = {}
        if self.config.atom_edge_interaction:
            bases_a2e["rad"] = self.mlp_rbf_aeint(basis_rad_a2ee2a_raw)
            bases_a2e["cir"] = list(
                self.mlp_cbf_aeint(
                    rad_basis=basis_rad_cir_a2e_raw,
                    sph_basis=basis_cir_a2e_raw,
                    idx_sph_outer=trip_idx_a2e["out"],
                    idx_sph_inner=trip_idx_a2e["out_agg"],
                )
            )
        bases_e2a: GraphBases = {}
        if self.config.edge_atom_interaction:
            bases_e2a["rad"] = self.mlp_rbf_eaint(basis_rad_main_raw)
            bases_e2a["cir"] = list(
                self.mlp_cbf_eaint(
                    rad_basis=basis_rad_cir_e2a_raw,
                    sph_basis=basis_cir_e2a_raw,
                    idx_rad_outer=a2ee2a_graph["edge_index"][1],
                    idx_rad_inner=a2ee2a_graph["target_neighbor_idx"],
                    idx_sph_outer=trip_idx_e2a["out"],
                    idx_sph_inner=trip_idx_e2a["out_agg"],
                    num_atoms=num_atoms,
                )
            )
        if self.config.atom_interaction:
            basis_a2a_rad = self.mlp_rbf_aint(
                rad_basis=basis_rad_a2a_raw,
                idx_rad_outer=a2a_graph["edge_index"][1],
                idx_rad_inner=a2a_graph["target_neighbor_idx"],
                num_atoms=num_atoms,
            )
        else:
            basis_a2a_rad = None

        bases_e2e: GraphBases = {}
        bases_e2e["rad"] = self.mlp_rbf_tint(basis_rad_main_raw)
        bases_e2e["cir"] = list(
            self.mlp_cbf_tint(
                rad_basis=basis_rad_cir_e2e_raw,
                sph_basis=basis_cir_e2e_raw,
                idx_sph_outer=trip_idx_e2e["out"],
                idx_sph_inner=trip_idx_e2e["out_agg"],
            )
        )

        basis_atom_update = self.mlp_rbf_h(basis_rad_main_raw)
        basis_output = self.mlp_rbf_out(basis_rad_main_raw)

        m = self.edge_emb(h, basis_rad_main_raw, main_graph["edge_index"])

        return BasesOutput(
            m,  # (e_main, emb_size_edge)
            basis_atom_update,  # (e_main, emb_size_rbf)
            basis_output,  # (e_main, emb_size_rbf)
            bases_qint,  # rad=(e_main, emb_size_rbf), cir=(num_triplets_qint, emb_size_cbf), sph=[(e_main, emb_size_sbf, num_spherical**2), (e_main, num_spherical**2, n_atoms)]
            bases_e2e,  # rad=(e_main, emb_size_rbf), cir=[(e_main, emb_size_sbf, num_spherical), (e_main, num_spherical, Kmax_e2e)]
            bases_a2e,  # rad=(e_a2ee2a, emb_size_rbf), cir=[(e_main, emb_size_sbf, num_spherical), (e_main, num_spherical, Kmax_a2e)]
            bases_e2a,  # rad=(e_main, emb_size_rbf), cir=[(n_atoms, emb_size_sbf, emb_size_interim), (e_a2ee2a, num_spherical, Kmax_e2e)]
            basis_a2a_rad,  # (n_atoms, emb_size_rbf, emb_size_interm)
        )


class TaskSpecificBases(nn.Module):
    def __init__(self, task_configs: list[BasesConfig]):
        super().__init__()

        self.bases_layers = TypedModuleList(
            [Bases(config) for config in task_configs]
        )

        self.shared_parameters: list[tuple[nn.Parameter, int]] = []
        for layer in self.bases_layers:
            self.shared_parameters.extend(layer.shared_parameters)

    def forward(
        self,
        data: Batch,
        *,
        h: torch.Tensor,
        main_graph: dict,
        a2a_graph: dict,
        a2ee2a_graph: dict,
        qint_graph: dict,
        trip_idx_e2e: dict,
        trip_idx_a2e: dict,
        trip_idx_e2a: dict,
        quad_idx: dict,
        num_atoms: int,
    ):
        # we now select the correct basis functions for the current task (data.task_idx)
        task_idx: torch.Tensor = data.task_idx  # (n_graphs,)
        task_idx_node = task_idx[data.batch]  # (n_nodes,)
        task_idx_main_graph = task_idx_node[
            main_graph["edge_index"][0]
        ]  # (n_edges_main,)
        task_idx_triplets_qint = task_idx_node[
            main_graph["edge_index"][0, quad_idx["triplet_in"]["in"]]
        ]
        task_idx_a2ee2a_graph = task_idx_node[a2ee2a_graph["edge_index"][0]]

        bases: BasesOutput | None = None
        for idx, bases_layer in enumerate(self.bases_layers):
            bases_task: BasesOutput = bases_layer(
                data,
                h=h,
                main_graph=main_graph,
                a2a_graph=a2a_graph,
                a2ee2a_graph=a2ee2a_graph,
                qint_graph=qint_graph,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
                num_atoms=num_atoms,
            )
            if bases is None:
                bases = bases_task

            task_mask_node = (task_idx_node == idx).unsqueeze(
                -1
            )  # (n_nodes, 1)
            task_mask_main_graph = (task_idx_main_graph == idx).unsqueeze(
                -1
            )  # (n_edges_main, 1)

            task_mask_triplets_qint = (
                task_idx_triplets_qint == idx
            ).unsqueeze(
                -1
            )  # (n_triplets_qint, 1)
            task_mask_a2ee2a_graph = (task_idx_a2ee2a_graph == idx).unsqueeze(
                -1
            )  # (n_edges_a2ee2a, 1)

            bases.m += task_mask_main_graph * bases_task.m
            bases.atom_update += task_mask_main_graph * bases_task.atom_update
            bases.output += task_mask_main_graph * bases_task.output

            bases.qint["rad"] += task_mask_main_graph * bases_task.qint["rad"]
            bases.qint["cir"] += (
                task_mask_triplets_qint * bases_task.qint["cir"]
            )
            bases.qint["sph"][0] += (
                task_mask_main_graph.unsqueeze(-1) * bases_task.qint["sph"][0]
            )
            bases.qint["sph"][1] += (
                task_mask_main_graph.unsqueeze(-1) * bases_task.qint["sph"][1]
            )

            bases.e2e["rad"] += task_mask_main_graph * bases_task.e2e["rad"]
            bases.e2e["cir"][0] += (
                task_mask_main_graph.unsqueeze(-1) * bases_task.e2e["cir"][0]
            )
            bases.e2e["cir"][1] += (
                task_mask_main_graph.unsqueeze(-1) * bases_task.e2e["cir"][1]
            )

            bases.a2e["rad"] += task_mask_a2ee2a_graph * bases_task.a2e["rad"]
            bases.a2e["cir"][0] += (
                task_mask_main_graph.unsqueeze(-1) * bases_task.a2e["cir"][0]
            )
            bases.a2e["cir"][1] += (
                task_mask_main_graph.unsqueeze(-1) * bases_task.a2e["cir"][1]
            )

            bases.e2a["rad"] += task_mask_main_graph * bases_task.e2a["rad"]
            bases.e2a["cir"][0] += (
                task_mask_node.unsqueeze(-1) * bases_task.e2a["cir"][0]
            )
            bases.e2a["cir"][1] += (
                task_mask_a2ee2a_graph.unsqueeze(-1) * bases_task.e2a["cir"][1]
            )

            if bases.a2a_rad is not None:
                assert bases_task.a2a_rad is not None
                bases.a2a_rad += (
                    task_mask_node.unsqueeze(-1) * bases_task.a2a_rad
                )

        return bases
