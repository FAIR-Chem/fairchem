"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from fm.datasets.utils.goc_graph import graphs_from_batch
from torch_geometric.data import Batch
from torch_scatter import segment_coo
from typing_extensions import override

from ocpmodels.common.registry import registry
from ocpmodels.common.typing import TypedModuleList
from ocpmodels.models.base import BaseModel
from ocpmodels.modules.scaling.compat import load_scales_compat

from .bases import Bases, BasesOutput
from .config import BackboneConfig, BasesConfig
from .interaction_indices import (
    get_mixed_triplets,
    get_quadruplets,
    get_triplets,
)
from .layers.atom_update_block import OutputBlock
from .layers.base_layers import Dense, ResidualLayer
from .layers.force_scaler import ForceScaler
from .layers.interaction_block import InteractionBlock
from .utils import (
    get_angle,
    get_edge_id,
    get_inner_idx,
    inner_product_clamped,
    repeat_blocks,
)


class FinalMLP(nn.Module):
    def __init__(
        self,
        *,
        emb_size: int,
        num_blocks: int,
        num_global_out_layers: int,
        activation: Optional[str] = None,
    ):
        super().__init__()

        out_mlp = [
            Dense(
                emb_size * (num_blocks + 1),
                emb_size,
                activation=activation,
            )
        ]
        out_mlp += [
            ResidualLayer(
                emb_size,
                activation=activation,
            )
            for _ in range(num_global_out_layers)
        ]
        self.out_mlp = nn.Sequential(*out_mlp)

    def forward(
        self,
        x: torch.Tensor,
        *,
        data: Batch,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.out_mlp(x)


class TaskSpecificFinalMLP(nn.Module, ABC):
    def __init__(
        self,
        *,
        n_tasks: int,
        emb_size: int,
        num_blocks: int,
        num_global_out_layers: int,
        activation: Optional[str] = None,
    ):
        super().__init__()

        self.outputs = TypedModuleList(
            [
                FinalMLP(
                    emb_size=emb_size,
                    num_blocks=num_blocks,
                    num_global_out_layers=num_global_out_layers,
                    activation=activation,
                )
                for _ in range(n_tasks)
            ]
        )

    @abstractmethod
    def create_task_idx(
        self,
        data: Batch,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def forward(
        self,
        x: torch.Tensor,
        *,
        data: Batch,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        task_idx: torch.Tensor = self.create_task_idx(data, edge_index)

        out: torch.Tensor = 0.0
        for i, output in enumerate(self.outputs):
            out += output(x, data=data, edge_index=edge_index) * (
                task_idx == i
            ).unsqueeze(-1)
        return out


class TaskSpecificEnergyFinalMLP(TaskSpecificFinalMLP):
    @override
    def create_task_idx(
        self,
        data: Batch,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        task_idx: torch.Tensor = data.task_idx  # b
        task_idx = task_idx[data.batch]  # n
        return task_idx


class TaskSpecificForcesFinalMLP(TaskSpecificFinalMLP):
    @override
    def create_task_idx(
        self,
        data: Batch,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        task_idx: torch.Tensor = data.task_idx  # b
        task_idx = task_idx[data.batch]  # n
        task_idx = task_idx[edge_index[0]]  # e
        return task_idx


@registry.register_model("gemnet_oc_mt")
class GemNetOCBackbone(BaseModel):

    """
    Arguments
    ---------
    num_atoms (int): Unused argument
    bond_feat_dim (int): Unused argument
    num_targets: int
        Number of prediction targets.

    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    num_blocks: int
        Number of building blocks to be stacked.

    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_trip_in: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        before the bilinear layer.
    emb_size_trip_out: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        after the bilinear layer.
    emb_size_quad_in: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        before the bilinear layer.
    emb_size_quad_out: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        after the bilinear layer.
    emb_size_aint_in: int
        Embedding size in the atom interaction before the bilinear layer.
    emb_size_aint_out: int
        Embedding size in the atom interaction after the bilinear layer.
    emb_size_rbf: int
        Embedding size of the radial basis transformation.
    emb_size_cbf: int
        Embedding size of the circular basis transformation (one angle).
    emb_size_sbf: int
        Embedding size of the spherical basis transformation (two angles).

    num_before_skip: int
        Number of residual blocks before the first skip connection.
    num_after_skip: int
        Number of residual blocks after the first skip connection.
    num_concat: int
        Number of residual blocks after the concatenation.
    num_atom: int
        Number of residual blocks in the atom embedding blocks.
    num_output_afteratom: int
        Number of residual blocks in the output blocks
        after adding the atom embedding.
    num_atom_emb_layers: int
        Number of residual blocks for transforming atom embeddings.
    num_global_out_layers: int
        Number of final residual blocks before the output.

    regress_forces: bool
        Whether to predict forces. Default: True
    direct_forces: bool
        If True predict forces based on aggregation of interatomic directions.
        If False predict forces based on negative gradient of energy potential.
    use_pbc: bool
        Whether to use periodic boundary conditions.
    scale_backprop_forces: bool
        Whether to scale up the energy and then scales down the forces
        to prevent NaNs and infs in backpropagated forces.

    rbf: dict
        Name and hyperparameters of the radial basis function.
    rbf_spherical: dict
        Name and hyperparameters of the radial basis function used as part of the
        circular and spherical bases.
        Optional. Uses rbf per default.
    envelope: dict
        Name and hyperparameters of the envelope function.
    cbf: dict
        Name and hyperparameters of the circular basis function.
    sbf: dict
        Name and hyperparameters of the spherical basis function.
    extensive: bool
        Whether the output should be extensive (proportional to the number of atoms)
    forces_coupled: bool
        If True, enforce that |F_st| = |F_ts|. No effect if direct_forces is False.
    output_init: str
        Initialization method for the final dense layer.
    activation: str
        Name of the activation function.
    scale_file: str
        Path to the pytorch file containing the scaling factors.

    quad_interaction: bool
        Whether to use quadruplet interactions (with dihedral angles)
    atom_edge_interaction: bool
        Whether to use atom-to-edge interactions
    edge_atom_interaction: bool
        Whether to use edge-to-atom interactions
    atom_interaction: bool
        Whether to use atom-to-atom interactions

    scale_basis: bool
        Whether to use a scaling layer in the raw basis function for better
        numerical stability.
    qint_tags: list
        Which atom tags to use quadruplet interactions for.
        0=sub-surface bulk, 1=surface, 2=adsorbate atoms.
    """

    qint_tags: torch.Tensor

    def __init__(
        self,
        output_targets: dict,
        config: BackboneConfig,
        *,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip_in: int,
        emb_size_trip_out: int,
        emb_size_quad_in: int,
        emb_size_quad_out: int,
        emb_size_aint_in: int,
        emb_size_aint_out: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        num_output_afteratom: int,
        num_atom_emb_layers: int = 0,
        num_global_out_layers: int = 2,
        regress_energy: bool = True,
        regress_forces: bool = True,
        direct_forces: bool = False,
        use_pbc: bool = True,
        scale_backprop_forces: bool = False,
        rbf: dict = {"name": "gaussian"},
        rbf_spherical: Optional[dict] = None,
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        sbf: dict = {"name": "spherical_harmonics"},
        extensive: bool = True,
        forces_coupled: bool = False,
        activation: str = "silu",
        quad_interaction: bool = False,
        atom_edge_interaction: bool = False,
        edge_atom_interaction: bool = False,
        atom_interaction: bool = False,
        scale_basis: bool = False,
        qint_tags: list = [0, 1, 2],
        num_elements: int = 120,
        otf_graph: bool = False,
        ln: bool = False,
        scale_file: Optional[str] = None,
        absolute_rbf_cutoff: float | None = None,
        edge_dropout: float | None = None,
        **kwargs,
    ):
        super().__init__(
            output_targets=output_targets,
            node_embedding_dim=emb_size_atom,
            edge_embedding_dim=emb_size_edge,
        )

        self.shared_parameters: list[tuple[nn.Parameter, int]] = []

        print("Unrecognized arguments: ", kwargs.keys())

        BackboneConfig.register(config)
        self.config = config

        if isinstance(self.config.ln, str) and self.config.ln not in (
            "pre",
            "post",
        ):
            raise ValueError(
                f"ln must be bool or 'pre' or 'post' but got {self.config.ln}"
            )

        self.ln = ln
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.extensive = extensive
        self.num_elements = num_elements

        self.atom_edge_interaction = atom_edge_interaction
        self.edge_atom_interaction = edge_atom_interaction
        self.atom_interaction = atom_interaction
        self.quad_interaction = quad_interaction
        self.otf_graph = otf_graph

        self.register_buffer(
            "qint_tags", torch.tensor(qint_tags), persistent=False
        )

        if not rbf_spherical:
            rbf_spherical = rbf

        self.use_pbc = use_pbc

        self.direct_forces = direct_forces
        self.forces_coupled = forces_coupled
        self.regress_forces = regress_forces
        self.regress_energy = regress_energy
        self.force_scaler = ForceScaler(enabled=scale_backprop_forces)

        self.bases = Bases(BasesConfig.from_backbone_config(self.config))
        if not self.config.unique_basis_per_layer:
            self.shared_parameters.extend(self.bases.shared_parameters)
        else:
            self.per_layer_bases = TypedModuleList(
                [
                    Bases(BasesConfig.from_backbone_config(self.config))
                    for _ in range(self.num_blocks)
                ]
            )

        # Embedding blocks
        # self.atom_emb = AtomEmbedding(emb_size_atom, num_elements)
        # self.edge_emb = EdgeEmbedding(
        #     emb_size_atom, num_radial, emb_size_edge, activation=activation
        # )
        self.atom_embedding = nn.Embedding(num_elements, emb_size_atom)

        # Interaction Blocks
        int_blocks = []
        for _ in range(num_blocks):
            int_blocks.append(
                InteractionBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip_in=emb_size_trip_in,
                    emb_size_trip_out=emb_size_trip_out,
                    emb_size_quad_in=emb_size_quad_in,
                    emb_size_quad_out=emb_size_quad_out,
                    emb_size_a2a_in=emb_size_aint_in,
                    emb_size_a2a_out=emb_size_aint_out,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_sbf=emb_size_sbf,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    num_atom_emb_layers=num_atom_emb_layers,
                    quad_interaction=quad_interaction,
                    atom_edge_interaction=atom_edge_interaction,
                    edge_atom_interaction=edge_atom_interaction,
                    atom_interaction=atom_interaction,
                    activation=activation,
                )
            )
        self.int_blocks = nn.ModuleList(int_blocks)
        out_blocks = []
        for _ in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    nHidden_afteratom=num_output_afteratom,
                    activation=activation,
                    direct_forces=direct_forces,
                    edge_dropout=edge_dropout,
                )
            )
        self.out_blocks = nn.ModuleList(out_blocks)

        # out_mlp_E = [
        #     Dense(
        #         emb_size_atom * (num_blocks + 1),
        #         emb_size_atom,
        #         activation=activation,
        #     )
        # ]
        # out_mlp_E += [
        #     ResidualLayer(
        #         emb_size_atom,
        #         activation=activation,
        #     )
        #     for _ in range(num_global_out_layers)
        # ]
        # self.out_mlp_E = nn.Sequential(*out_mlp_E)
        self.out_mlp_E = FinalMLP(
            emb_size=emb_size_atom,
            num_blocks=num_blocks,
            num_global_out_layers=num_global_out_layers,
            activation=activation,
        )
        if direct_forces:
            # out_mlp_F = [
            #     Dense(
            #         emb_size_edge * (num_blocks + 1),
            #         emb_size_edge,
            #         activation=activation,
            #     )
            # ]
            # out_mlp_F += [
            #     ResidualLayer(
            #         emb_size_edge,
            #         activation=activation,
            #     )
            #     for _ in range(num_global_out_layers)
            # ]
            # self.out_mlp_F = nn.Sequential(*out_mlp_F)
            self.out_mlp_F = FinalMLP(
                emb_size=emb_size_edge,
                num_blocks=num_blocks,
                num_global_out_layers=num_global_out_layers,
                activation=activation,
            )

        load_scales_compat(self, scale_file)

    def calculate_quad_angles(
        self,
        V_st,
        V_qint_st,
        quad_idx,
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

    def select_symmetric_edges(self, tensor, mask, reorder_idx, opposite_neg):
        """Use a mask to remove values of removed edges and then
        duplicate the values for the correct edge direction.

        Arguments
        ---------
        tensor: torch.Tensor
            Values to symmetrize for the new tensor.
        mask: torch.Tensor
            Mask defining which edges go in the correct direction.
        reorder_idx: torch.Tensor
            Indices defining how to reorder the tensor values after
            concatenating the edge values of both directions.
        opposite_neg: bool
            Whether the edge in the opposite direction should use the
            negative tensor value.

        Returns
        -------
        tensor_ordered: torch.Tensor
            A tensor with symmetrized values.
        """
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * opposite_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def symmetrize_edges(
        self,
        graph,
        batch_idx,
    ):
        """
        Symmetrize edges to ensure existence of counter-directional edges.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors.
        We only use i->j edges here. So we lose some j->i edges
        and add others by making it symmetric.
        """
        num_atoms = batch_idx.shape[0]
        new_graph = {}

        # Generate mask
        mask_sep_atoms = graph["edge_index"][0] < graph["edge_index"][1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (graph["cell_offset"][:, 0] < 0)
            | (
                (graph["cell_offset"][:, 0] == 0)
                & (graph["cell_offset"][:, 1] < 0)
            )
            | (
                (graph["cell_offset"][:, 0] == 0)
                & (graph["cell_offset"][:, 1] == 0)
                & (graph["cell_offset"][:, 2] < 0)
            )
        )
        mask_same_atoms = graph["edge_index"][0] == graph["edge_index"][1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_directed = graph["edge_index"][
            mask[None, :].expand(2, -1)
        ].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [edge_index_directed, edge_index_directed.flip(0)],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(
                graph["num_neighbors"].size(0),
                device=graph["edge_index"].device,
            ),
            graph["num_neighbors"],
        )
        batch_edge = batch_edge[mask]
        # segment_coo assumes sorted batch_edge
        # Factor 2 since this is only one half of the edges
        ones = batch_edge.new_ones(1).expand_as(batch_edge)
        new_graph["num_neighbors"] = 2 * segment_coo(
            ones, batch_edge, dim_size=graph["num_neighbors"].size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            torch.div(new_graph["num_neighbors"], 2, rounding_mode="floor"),
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_directed.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        new_graph["edge_index"] = edge_index_cat[:, edge_reorder_idx]
        new_graph["cell_offset"] = self.select_symmetric_edges(
            graph["cell_offset"], mask, edge_reorder_idx, True
        )
        new_graph["distance"] = self.select_symmetric_edges(
            graph["distance"], mask, edge_reorder_idx, False
        )
        new_graph["vector"] = self.select_symmetric_edges(
            graph["vector"], mask, edge_reorder_idx, True
        )

        # Indices for swapping c->a and a->c (for symmetric MP)
        # To obtain these efficiently and without any index assumptions,
        # we get order the counter-edge IDs and then
        # map this order back to the edge IDs.
        # Double argsort gives the desired mapping
        # from the ordered tensor to the original tensor.
        edge_ids = get_edge_id(
            new_graph["edge_index"], new_graph["cell_offset"], num_atoms
        )
        order_edge_ids = torch.argsort(edge_ids)
        inv_order_edge_ids = torch.argsort(order_edge_ids)
        edge_ids_counter = get_edge_id(
            new_graph["edge_index"].flip(0),
            -new_graph["cell_offset"],
            num_atoms,
        )
        order_edge_ids_counter = torch.argsort(edge_ids_counter)
        id_swap = order_edge_ids_counter[inv_order_edge_ids]

        return new_graph, id_swap

    def get_graphs_and_indices(self, data):
        """ "Generate embedding and interaction graphs and indices."""
        num_atoms = data.atomic_numbers.size(0)
        assert (
            self.atom_edge_interaction
            and self.edge_atom_interaction
            and self.atom_interaction
            and self.quad_interaction
        ), "Only the full interaction graph (ae + ea + a + q) is supported."

        graphs = graphs_from_batch(data)
        a2a_graph = graphs["a2a"]
        a2ee2a_graph = graphs["a2ee2a"]
        main_graph = graphs["main"]
        qint_graph = graphs["qint"]

        # Symmetrize edges for swapping in symmetric message passing
        if True:
            main_graph, id_swap = self.symmetrize_edges(main_graph, data.batch)
        else:
            raise NotImplementedError
            id_swap = main_graph.get("id_swap_edge_index", None)
            if id_swap is None:
                raise ValueError(
                    "Expected id_swap in main_graph for symmetric MP, but it was not found."
                )

        trip_idx_e2e = get_triplets(main_graph, num_atoms=num_atoms)

        # Additional indices for quadruplets
        if self.quad_interaction:
            quad_idx = get_quadruplets(
                main_graph,
                qint_graph,
                num_atoms,
            )
        else:
            quad_idx = {}

        if self.atom_edge_interaction:
            trip_idx_a2e = get_mixed_triplets(
                a2ee2a_graph,
                main_graph,
                num_atoms=num_atoms,
                return_agg_idx=True,
            )
        else:
            trip_idx_a2e = {}
        if self.edge_atom_interaction:
            trip_idx_e2a = get_mixed_triplets(
                main_graph,
                a2ee2a_graph,
                num_atoms=num_atoms,
                return_agg_idx=True,
            )
            # a2ee2a_graph['edge_index'][1] has to be sorted for this
            a2ee2a_graph["target_neighbor_idx"] = get_inner_idx(
                a2ee2a_graph["edge_index"][1], dim_size=num_atoms
            )
        else:
            trip_idx_e2a = {}
        if self.atom_interaction:
            # a2a_graph['edge_index'][1] has to be sorted for this
            a2a_graph["target_neighbor_idx"] = get_inner_idx(
                a2a_graph["edge_index"][1], dim_size=num_atoms
            )

        return (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        )

    @override
    def _forward(self, data: Batch):
        pos = data.pos
        # batch = data.batch
        # atomic_numbers = data.atomic_numbers.long()
        num_atoms = data.atomic_numbers.shape[0]

        # ActSave(
        #     h_initial=h,
        #     atomic_numbers=data.atomic_numbers,
        #     pos=pos,
        # )

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        ) = self.get_graphs_and_indices(data)
        idx_s, idx_t = main_graph["edge_index"]

        # ActSave(edge_index=main_graph["edge_index"])

        # Embedding block
        h = self.atom_embedding(data.atomic_numbers - 1)

        bases: BasesOutput = self.bases(
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
        m = bases.m
        # ActSave(m_initial=m)

        # (nAtoms, emb_size_atom)
        # m = self.edge_emb(h, bases.rbf_main, main_graph["edge_index"])
        # (nEdges_main, emb_size_edge)

        x_E, x_F = self.out_blocks[0](h, m, bases.output, idx_t, data=data)
        # ActSave({"x_E_0": x_E, "x_F_0": x_F})
        # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
        xs_E, xs_F = [x_E], [x_F]

        for i in range(self.num_blocks):
            # ActSave({f"h_{i}": h, f"m_{i}": m})
            if self.config.unique_basis_per_layer:
                bases: BasesOutput = self.per_layer_bases[i](
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
                m = m + bases.m

            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                bases_qint=bases.qint,
                bases_e2e=bases.e2e,
                bases_a2e=bases.a2e,
                bases_e2a=bases.e2a,
                basis_a2a_rad=bases.a2a_rad,
                basis_atom_update=bases.atom_update,
                edge_index_main=main_graph["edge_index"],
                a2ee2a_graph=a2ee2a_graph,
                a2a_graph=a2a_graph,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            x_E, x_F = self.out_blocks[i + 1](
                h, m, bases.output, idx_t, data=data
            )
            # ActSave({f"x_E_{i+1}": x_E, f"x_F_{i+1}": x_F})
            # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
            xs_E.append(x_E)
            xs_F.append(x_F)

        # ActSave({f"h_{self.num_blocks}": h, f"m_{self.num_blocks}": m})

        # Global output block for final predictions
        x_E = self.out_mlp_E(
            torch.cat(xs_E, dim=-1),
            data=data,
            edge_index=main_graph["edge_index"],
        )

        if self.direct_forces:
            x_F = self.out_mlp_F(
                torch.cat(xs_F, dim=-1),
                data=data,
                edge_index=main_graph["edge_index"],
            )
        else:
            x_F = None

        outputs = {
            "edge_idx": idx_t,
            "edge_vec": main_graph["vector"],
            "node_embedding": x_E,
            "edge_embedding": x_F,
        }
        return outputs
