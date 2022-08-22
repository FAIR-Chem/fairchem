"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math

import torch

from .atom_update_block import AtomUpdateBlock
from .base_layers import Dense, ResidualLayer
from .efficient import EfficientInteractionBilinear
from .embedding_block import EdgeEmbedding
from .scaling import ScaledModule, ScalingFactor


class InteractionBlock(ScaledModule):
    """
    Interaction block for GemNet-Q/dQ.

    Arguments
    ---------
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
    emb_size_a2a_in: int
        Embedding size in the atom interaction before the bilinear layer.
    emb_size_a2a_out: int
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
    num_atom_emb_layers: int
        Number of residual blocks for transforming atom embeddings.
    quad_interaction: bool
        Whether to use quadruplet interactions.
    atom_edge_interaction: bool
        Whether to use atom-to-edge interactions.
    edge_atom_interaction: bool
        Whether to use edge-to-atom interactions.
    atom_interaction: bool
        Whether to use atom-to-atom interactions.
    activation: str
        Name of the activation function to use in the dense layers.
    """

    def __init__(
        self,
        emb_size_atom,
        emb_size_edge,
        emb_size_trip_in,
        emb_size_trip_out,
        emb_size_quad_in,
        emb_size_quad_out,
        emb_size_a2a_in,
        emb_size_a2a_out,
        emb_size_rbf,
        emb_size_cbf,
        emb_size_sbf,
        num_before_skip,
        num_after_skip,
        num_concat,
        num_atom,
        num_atom_emb_layers=0,
        quad_interaction=False,
        atom_edge_interaction=False,
        edge_atom_interaction=False,
        atom_interaction=False,
        activation=None,
    ):
        super().__init__()

        ## ------------------------ Message Passing ----------------------- ##
        # Dense transformation of skip connection
        self.dense_ca = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Triplet Interaction
        self.trip_interaction = TripletInteraction(
            emb_size_in=emb_size_edge,
            emb_size_out=emb_size_edge,
            emb_size_trip_in=emb_size_trip_in,
            emb_size_trip_out=emb_size_trip_out,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            symmetric_mp=True,
            swap_output=True,
            activation=activation,
        )

        # Quadruplet Interaction
        if quad_interaction:
            self.quad_interaction = QuadrupletInteraction(
                emb_size_edge=emb_size_edge,
                emb_size_quad_in=emb_size_quad_in,
                emb_size_quad_out=emb_size_quad_out,
                emb_size_rbf=emb_size_rbf,
                emb_size_cbf=emb_size_cbf,
                emb_size_sbf=emb_size_sbf,
                symmetric_mp=True,
                activation=activation,
            )
        else:
            self.quad_interaction = None

        if atom_edge_interaction:
            self.atom_edge_interaction = TripletInteraction(
                emb_size_in=emb_size_atom,
                emb_size_out=emb_size_edge,
                emb_size_trip_in=emb_size_trip_in,
                emb_size_trip_out=emb_size_trip_out,
                emb_size_rbf=emb_size_rbf,
                emb_size_cbf=emb_size_cbf,
                symmetric_mp=True,
                swap_output=True,
                activation=activation,
            )
        else:
            self.atom_edge_interaction = None
        if edge_atom_interaction:
            self.edge_atom_interaction = TripletInteraction(
                emb_size_in=emb_size_edge,
                emb_size_out=emb_size_atom,
                emb_size_trip_in=emb_size_trip_in,
                emb_size_trip_out=emb_size_trip_out,
                emb_size_rbf=emb_size_rbf,
                emb_size_cbf=emb_size_cbf,
                symmetric_mp=False,
                swap_output=False,
                activation=activation,
            )
        else:
            self.edge_atom_interaction = None
        if atom_interaction:
            self.atom_interaction = PairInteraction(
                emb_size_atom=emb_size_atom,
                emb_size_pair_in=emb_size_a2a_in,
                emb_size_pair_out=emb_size_a2a_out,
                emb_size_rbf=emb_size_rbf,
                activation=activation,
            )
        else:
            self.atom_interaction = None

        ## -------------------- Update Edge Embeddings -------------------- ##
        # Residual layers before skip connection
        self.layers_before_skip = torch.nn.ModuleList(
            [
                ResidualLayer(
                    emb_size_edge,
                    activation=activation,
                )
                for i in range(num_before_skip)
            ]
        )

        # Residual layers after skip connection
        self.layers_after_skip = torch.nn.ModuleList(
            [
                ResidualLayer(
                    emb_size_edge,
                    activation=activation,
                )
                for i in range(num_after_skip)
            ]
        )

        ## -------------------- Update Atom Embeddings -------------------- ##
        self.atom_emb_layers = torch.nn.ModuleList(
            [
                ResidualLayer(
                    emb_size_atom,
                    activation=activation,
                )
                for _ in range(num_atom_emb_layers)
            ]
        )

        self.atom_update = AtomUpdateBlock(
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            nHidden=num_atom,
            activation=activation,
        )

        ## ---------- Update Edge Embeddings with Atom Embeddings --------- ##
        self.concat_layer = EdgeEmbedding(
            emb_size_atom,
            emb_size_edge,
            emb_size_edge,
            activation=activation,
        )
        self.residual_m = torch.nn.ModuleList(
            [
                ResidualLayer(emb_size_edge, activation=activation)
                for _ in range(num_concat)
            ]
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        num_eint = 2.0 + quad_interaction + atom_edge_interaction
        self.inv_sqrt_num_eint = 1 / math.sqrt(num_eint)
        num_aint = 1.0 + edge_atom_interaction + atom_interaction
        self.inv_sqrt_num_aint = 1 / math.sqrt(num_aint)

    def forward(
        self,
        h,
        m,
        bases_qint,
        bases_e2e,
        bases_a2e,
        bases_e2a,
        basis_a2a_rad,
        basis_atom_update,
        edge_index_main,
        a2ee2a_graph,
        a2a_graph,
        id_swap,
        trip_idx_e2e,
        trip_idx_a2e,
        trip_idx_e2a,
        quad_idx,
    ):
        """
        Returns
        -------
        h: torch.Tensor, shape=(nEdges, emb_size_atom)
            Atom embeddings.
        m: torch.Tensor, shape=(nEdges, emb_size_edge)
            Edge embeddings (c->a).
        """
        num_atoms = h.shape[0]

        # Initial transformation
        x_ca_skip = self.dense_ca(m)  # (nEdges, emb_size_edge)

        x_e2e = self.trip_interaction(
            m,
            bases_e2e,
            trip_idx_e2e,
            id_swap,
        )
        if self.quad_interaction is not None:
            x_qint = self.quad_interaction(
                m,
                bases_qint,
                quad_idx,
                id_swap,
            )
        if self.atom_edge_interaction is not None:
            x_a2e = self.atom_edge_interaction(
                h,
                bases_a2e,
                trip_idx_a2e,
                id_swap,
                expand_idx=a2ee2a_graph["edge_index"][0],
            )
        if self.edge_atom_interaction is not None:
            h_e2a = self.edge_atom_interaction(
                m,
                bases_e2a,
                trip_idx_e2a,
                id_swap,
                idx_agg2=a2ee2a_graph["edge_index"][1],
                idx_agg2_inner=a2ee2a_graph["target_neighbor_idx"],
                agg2_out_size=num_atoms,
            )
        if self.atom_interaction is not None:
            h_a2a = self.atom_interaction(
                h,
                basis_a2a_rad,
                a2a_graph["edge_index"],
                a2a_graph["target_neighbor_idx"],
            )

        ## -------------- Merge Embeddings after interactions ------------- ##
        x = x_ca_skip + x_e2e  # (nEdges, emb_size_edge)
        if self.quad_interaction is not None:
            x += x_qint  # (nEdges, emb_size_edge)
        if self.atom_edge_interaction is not None:
            x += x_a2e  # (nEdges, emb_size_edge)
        x = x * self.inv_sqrt_num_eint

        # Merge atom embeddings after interactions
        if self.edge_atom_interaction is not None:
            h = h + h_e2a  # (nEdges, emb_size_edge)
        if self.atom_interaction is not None:
            h = h + h_a2a  # (nEdges, emb_size_edge)
        h = h * self.inv_sqrt_num_aint

        ## -------------------- Update Edge Embeddings -------------------- ##
        # Transformations before skip connection
        for i, layer in enumerate(self.layers_before_skip):
            x = layer(x)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + x  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2

        # Transformations after skip connection
        for i, layer in enumerate(self.layers_after_skip):
            m = layer(m)  # (nEdges, emb_size_edge)

        ## -------------------- Update Atom Embeddings -------------------- ##
        for layer in self.atom_emb_layers:
            h = layer(h)  # (nAtoms, emb_size_atom)

        h2 = self.atom_update(h, m, basis_atom_update, edge_index_main[1])

        # Skip connection
        h = h + h2  # (nAtoms, emb_size_atom)
        h = h * self.inv_sqrt_2

        ## ---------- Update Edge Embeddings with Atom Embeddings --------- ##
        m2 = self.concat_layer(h, m, edge_index_main)
        # (nEdges, emb_size_edge)

        for i, layer in enumerate(self.residual_m):
            m2 = layer(m2)  # (nEdges, emb_size_edge)

        # Skip connection
        m = m + m2  # (nEdges, emb_size_edge)
        m = m * self.inv_sqrt_2
        return h, m


class QuadrupletInteraction(ScaledModule):
    """
    Quadruplet-based message passing block.

    Arguments
    ---------
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_quad_in: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        before the bilinear layer.
    emb_size_quad_out: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        after the bilinear layer.
    emb_size_rbf: int
        Embedding size of the radial basis transformation.
    emb_size_cbf: int
        Embedding size of the circular basis transformation (one angle).
    emb_size_sbf: int
        Embedding size of the spherical basis transformation (two angles).
    symmetric_mp: bool
        Whether to use symmetric message passing and
        update the edges in both directions.
    activation: str
        Name of the activation function to use in the dense layers.
    """

    def __init__(
        self,
        emb_size_edge,
        emb_size_quad_in,
        emb_size_quad_out,
        emb_size_rbf,
        emb_size_cbf,
        emb_size_sbf,
        symmetric_mp=True,
        activation=None,
    ):
        super().__init__()
        self.symmetric_mp = symmetric_mp

        # Dense transformation
        self.dense_db = Dense(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Up projections of basis representations,
        # bilinear layer and scaling factors
        self.mlp_rbf = Dense(
            emb_size_rbf,
            emb_size_edge,
            activation=None,
            bias=False,
        )
        self.scale_rbf = ScalingFactor()

        self.mlp_cbf = Dense(
            emb_size_cbf,
            emb_size_quad_in,
            activation=None,
            bias=False,
        )
        self.scale_cbf = ScalingFactor()

        self.mlp_sbf = EfficientInteractionBilinear(
            emb_size_quad_in, emb_size_sbf, emb_size_quad_out
        )
        self.scale_sbf_sum = ScalingFactor()
        # combines scaling for bilinear layer and summation

        # Down and up projections
        self.down_projection = Dense(
            emb_size_edge,
            emb_size_quad_in,
            activation=activation,
            bias=False,
        )
        self.up_projection_ca = Dense(
            emb_size_quad_out,
            emb_size_edge,
            activation=activation,
            bias=False,
        )
        if self.symmetric_mp:
            self.up_projection_ac = Dense(
                emb_size_quad_out,
                emb_size_edge,
                activation=activation,
                bias=False,
            )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        m,
        bases,
        idx,
        id_swap,
    ):
        """
        Returns
        -------
        m: torch.Tensor, shape=(nEdges, emb_size_edge)
            Edge embeddings (c->a).
        """

        x_db = self.dense_db(m)  # (nEdges, emb_size_edge)

        # Transform via radial basis
        x_db2 = x_db * self.mlp_rbf(bases["rad"])  # (nEdges, emb_size_edge)
        x_db = self.scale_rbf(x_db2, x_ref=x_db)

        # Down project embeddings
        x_db = self.down_projection(x_db)  # (nEdges, emb_size_quad_in)

        # Transform via circular basis
        x_db = x_db[idx["triplet_in"]["in"]]
        # (num_triplets_int, emb_size_quad_in)

        x_db2 = x_db * self.mlp_cbf(bases["cir"])
        # (num_triplets_int, emb_size_quad_in)
        x_db = self.scale_cbf(x_db2, x_ref=x_db)

        # Transform via spherical basis
        x_db = x_db[idx["trip_in_to_quad"]]
        # (num_quadruplets, emb_size_quad_in)
        x = self.mlp_sbf(bases["sph"], x_db, idx["out"], idx["out_agg"])
        # (nEdges, emb_size_quad_out)
        x = self.scale_sbf_sum(x, x_ref=x_db)

        # =>
        # rbf(d_db)
        # cbf(d_ba, angle_abd)
        # sbf(d_ca, angle_cab, angle_cabd)

        if self.symmetric_mp:
            # Upproject embeddings
            x_ca = self.up_projection_ca(x)  # (nEdges, emb_size_edge)
            x_ac = self.up_projection_ac(x)  # (nEdges, emb_size_edge)

            # Merge interaction of c->a and a->c
            x_ac = x_ac[id_swap]  # swap to add to edge a->c and not c->a
            x_res = x_ca + x_ac
            x_res = x_res * self.inv_sqrt_2
            return x_res
        else:
            x_res = self.up_projection_ca(x)
            return x_res


class TripletInteraction(ScaledModule):
    """
    Triplet-based message passing block.

    Arguments
    ---------
    emb_size_in: int
        Embedding size of the input embeddings.
    emb_size_out: int
        Embedding size of the output embeddings.
    emb_size_trip_in: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        before the bilinear layer.
    emb_size_trip_out: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        after the bilinear layer.
    emb_size_rbf: int
        Embedding size of the radial basis transformation.
    emb_size_cbf: int
        Embedding size of the circular basis transformation (one angle).
    symmetric_mp: bool
        Whether to use symmetric message passing and
        update the edges in both directions.
    swap_output: bool
        Whether to swap the output embedding directions.
        Only relevant if symmetric_mp is False.
    activation: str
        Name of the activation function to use in the dense layers.
    """

    def __init__(
        self,
        emb_size_in,
        emb_size_out,
        emb_size_trip_in,
        emb_size_trip_out,
        emb_size_rbf,
        emb_size_cbf,
        symmetric_mp=True,
        swap_output=True,
        activation=None,
    ):
        super().__init__()
        self.symmetric_mp = symmetric_mp
        self.swap_output = swap_output

        # Dense transformation
        self.dense_ba = Dense(
            emb_size_in,
            emb_size_in,
            activation=activation,
            bias=False,
        )

        # Up projections of basis representations, bilinear layer and scaling factors
        self.mlp_rbf = Dense(
            emb_size_rbf,
            emb_size_in,
            activation=None,
            bias=False,
        )
        self.scale_rbf = ScalingFactor()

        self.mlp_cbf = EfficientInteractionBilinear(
            emb_size_trip_in, emb_size_cbf, emb_size_trip_out
        )
        self.scale_cbf_sum = ScalingFactor()
        # combines scaling for bilinear layer and summation

        # Down and up projections
        self.down_projection = Dense(
            emb_size_in,
            emb_size_trip_in,
            activation=activation,
            bias=False,
        )
        self.up_projection_ca = Dense(
            emb_size_trip_out,
            emb_size_out,
            activation=activation,
            bias=False,
        )
        if self.symmetric_mp:
            self.up_projection_ac = Dense(
                emb_size_trip_out,
                emb_size_out,
                activation=activation,
                bias=False,
            )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        m,
        bases,
        idx,
        id_swap,
        expand_idx=None,
        idx_agg2=None,
        idx_agg2_inner=None,
        agg2_out_size=None,
    ):
        """
        Returns
        -------
        m: torch.Tensor, shape=(nEdges, emb_size_edge)
            Edge embeddings.
        """

        # Dense transformation
        x_ba = self.dense_ba(m)  # (nEdges, emb_size_edge)

        if expand_idx is not None:
            x_ba = x_ba[expand_idx]

        # Transform via radial basis
        rad_emb = self.mlp_rbf(bases["rad"])  # (nEdges, emb_size_edge)
        x_ba2 = x_ba * rad_emb
        x_ba = self.scale_rbf(x_ba2, x_ref=x_ba)

        x_ba = self.down_projection(x_ba)  # (nEdges, emb_size_trip_in)

        # Transform via circular spherical basis
        x_ba = x_ba[idx["in"]]

        # Efficient bilinear layer
        x = self.mlp_cbf(
            basis=bases["cir"],
            m=x_ba,
            idx_agg_outer=idx["out"],
            idx_agg_inner=idx["out_agg"],
            idx_agg2_outer=idx_agg2,
            idx_agg2_inner=idx_agg2_inner,
            agg2_out_size=agg2_out_size,
        )
        # (num_atoms, emb_size_trip_out)
        x = self.scale_cbf_sum(x, x_ref=x_ba)

        # =>
        # rbf(d_ba)
        # cbf(d_ca, angle_cab)

        if self.symmetric_mp:
            # Up project embeddings
            x_ca = self.up_projection_ca(x)  # (nEdges, emb_size_edge)
            x_ac = self.up_projection_ac(x)  # (nEdges, emb_size_edge)

            # Merge interaction of c->a and a->c
            x_ac = x_ac[id_swap]  # swap to add to edge a->c and not c->a
            x_res = x_ca + x_ac
            x_res = x_res * self.inv_sqrt_2
            return x_res
        else:
            if self.swap_output:
                x = x[id_swap]
            x_res = self.up_projection_ca(x)  # (nEdges, emb_size_edge)
            return x_res


class PairInteraction(ScaledModule):
    """
    Pair-based message passing block.

    Arguments
    ---------
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_pair_in: int
        Embedding size of the atom pairs before the bilinear layer.
    emb_size_pair_out: int
        Embedding size of the atom pairs after the bilinear layer.
    emb_size_rbf: int
        Embedding size of the radial basis transformation.
    activation: str
        Name of the activation function to use in the dense layers.
    """

    def __init__(
        self,
        emb_size_atom,
        emb_size_pair_in,
        emb_size_pair_out,
        emb_size_rbf,
        activation=None,
    ):
        super().__init__()

        # Bilinear layer and scaling factor
        self.bilinear = Dense(
            emb_size_rbf * emb_size_pair_in,
            emb_size_pair_out,
            activation=None,
            bias=False,
        )
        self.scale_rbf_sum = ScalingFactor()

        # Down and up projections
        self.down_projection = Dense(
            emb_size_atom,
            emb_size_pair_in,
            activation=activation,
            bias=False,
        )
        self.up_projection = Dense(
            emb_size_pair_out,
            emb_size_atom,
            activation=activation,
            bias=False,
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(
        self,
        h,
        rad_basis,
        edge_index,
        target_neighbor_idx,
    ):
        """
        Returns
        -------
        h: torch.Tensor, shape=(num_atoms, emb_size_atom)
            Atom embeddings.
        """
        num_atoms = h.shape[0]

        x_b = self.down_projection(h)  # (num_atoms, emb_size_edge)
        x_ba = x_b[edge_index[0]]  # (num_edges, emb_size_edge)

        Kmax = torch.max(target_neighbor_idx) + 1
        x2 = x_ba.new_zeros(num_atoms, Kmax, x_ba.shape[-1])
        x2[edge_index[1], target_neighbor_idx] = x_ba
        # (num_atoms, Kmax, emb_size_edge)

        x_ba2 = rad_basis @ x2
        # (num_atoms, emb_size_interm, emb_size_edge)
        h_out = self.bilinear(x_ba2.reshape(num_atoms, -1))

        h_out = self.scale_rbf_sum(h_out, x_ref=x_ba)
        # (num_atoms, emb_size_edge)

        h_out = self.up_projection(h_out)  # (num_atoms, emb_size_atom)

        return h_out
