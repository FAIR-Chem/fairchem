core.models.gemnet_oc.layers.interaction_block
==============================================

.. py:module:: core.models.gemnet_oc.layers.interaction_block

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_oc.layers.interaction_block.InteractionBlock
   core.models.gemnet_oc.layers.interaction_block.QuadrupletInteraction
   core.models.gemnet_oc.layers.interaction_block.TripletInteraction
   core.models.gemnet_oc.layers.interaction_block.PairInteraction


Module Contents
---------------

.. py:class:: InteractionBlock(emb_size_atom: int, emb_size_edge: int, emb_size_trip_in: int, emb_size_trip_out: int, emb_size_quad_in: int, emb_size_quad_out: int, emb_size_a2a_in: int, emb_size_a2a_out: int, emb_size_rbf: int, emb_size_cbf: int, emb_size_sbf: int, num_before_skip: int, num_after_skip: int, num_concat: int, num_atom: int, num_atom_emb_layers: int = 0, quad_interaction: bool = False, atom_edge_interaction: bool = False, edge_atom_interaction: bool = False, atom_interaction: bool = False, activation=None)

   Bases: :py:obj:`torch.nn.Module`


   Interaction block for GemNet-Q/dQ.

   :param emb_size_atom: Embedding size of the atoms.
   :type emb_size_atom: int
   :param emb_size_edge: Embedding size of the edges.
   :type emb_size_edge: int
   :param emb_size_trip_in: (Down-projected) embedding size of the quadruplet edge embeddings
                            before the bilinear layer.
   :type emb_size_trip_in: int
   :param emb_size_trip_out: (Down-projected) embedding size of the quadruplet edge embeddings
                             after the bilinear layer.
   :type emb_size_trip_out: int
   :param emb_size_quad_in: (Down-projected) embedding size of the quadruplet edge embeddings
                            before the bilinear layer.
   :type emb_size_quad_in: int
   :param emb_size_quad_out: (Down-projected) embedding size of the quadruplet edge embeddings
                             after the bilinear layer.
   :type emb_size_quad_out: int
   :param emb_size_a2a_in: Embedding size in the atom interaction before the bilinear layer.
   :type emb_size_a2a_in: int
   :param emb_size_a2a_out: Embedding size in the atom interaction after the bilinear layer.
   :type emb_size_a2a_out: int
   :param emb_size_rbf: Embedding size of the radial basis transformation.
   :type emb_size_rbf: int
   :param emb_size_cbf: Embedding size of the circular basis transformation (one angle).
   :type emb_size_cbf: int
   :param emb_size_sbf: Embedding size of the spherical basis transformation (two angles).
   :type emb_size_sbf: int
   :param num_before_skip: Number of residual blocks before the first skip connection.
   :type num_before_skip: int
   :param num_after_skip: Number of residual blocks after the first skip connection.
   :type num_after_skip: int
   :param num_concat: Number of residual blocks after the concatenation.
   :type num_concat: int
   :param num_atom: Number of residual blocks in the atom embedding blocks.
   :type num_atom: int
   :param num_atom_emb_layers: Number of residual blocks for transforming atom embeddings.
   :type num_atom_emb_layers: int
   :param quad_interaction: Whether to use quadruplet interactions.
   :type quad_interaction: bool
   :param atom_edge_interaction: Whether to use atom-to-edge interactions.
   :type atom_edge_interaction: bool
   :param edge_atom_interaction: Whether to use edge-to-atom interactions.
   :type edge_atom_interaction: bool
   :param atom_interaction: Whether to use atom-to-atom interactions.
   :type atom_interaction: bool
   :param activation: Name of the activation function to use in the dense layers.
   :type activation: str


   .. py:attribute:: dense_ca


   .. py:attribute:: trip_interaction


   .. py:attribute:: layers_before_skip


   .. py:attribute:: layers_after_skip


   .. py:attribute:: atom_emb_layers


   .. py:attribute:: atom_update


   .. py:attribute:: concat_layer


   .. py:attribute:: residual_m


   .. py:attribute:: inv_sqrt_2


   .. py:attribute:: inv_sqrt_num_eint


   .. py:attribute:: inv_sqrt_num_aint


   .. py:method:: forward(h, m, bases_qint, bases_e2e, bases_a2e, bases_e2a, basis_a2a_rad, basis_atom_update, edge_index_main, a2ee2a_graph, a2a_graph, id_swap, trip_idx_e2e, trip_idx_a2e, trip_idx_e2a, quad_idx)

      :returns: * **h** (*torch.Tensor, shape=(nEdges, emb_size_atom)*) -- Atom embeddings.
                * **m** (*torch.Tensor, shape=(nEdges, emb_size_edge)*) -- Edge embeddings (c->a).



.. py:class:: QuadrupletInteraction(emb_size_edge, emb_size_quad_in, emb_size_quad_out, emb_size_rbf, emb_size_cbf, emb_size_sbf, symmetric_mp=True, activation=None)

   Bases: :py:obj:`torch.nn.Module`


   Quadruplet-based message passing block.

   :param emb_size_edge: Embedding size of the edges.
   :type emb_size_edge: int
   :param emb_size_quad_in: (Down-projected) embedding size of the quadruplet edge embeddings
                            before the bilinear layer.
   :type emb_size_quad_in: int
   :param emb_size_quad_out: (Down-projected) embedding size of the quadruplet edge embeddings
                             after the bilinear layer.
   :type emb_size_quad_out: int
   :param emb_size_rbf: Embedding size of the radial basis transformation.
   :type emb_size_rbf: int
   :param emb_size_cbf: Embedding size of the circular basis transformation (one angle).
   :type emb_size_cbf: int
   :param emb_size_sbf: Embedding size of the spherical basis transformation (two angles).
   :type emb_size_sbf: int
   :param symmetric_mp: Whether to use symmetric message passing and
                        update the edges in both directions.
   :type symmetric_mp: bool
   :param activation: Name of the activation function to use in the dense layers.
   :type activation: str


   .. py:attribute:: symmetric_mp


   .. py:attribute:: dense_db


   .. py:attribute:: mlp_rbf


   .. py:attribute:: scale_rbf


   .. py:attribute:: mlp_cbf


   .. py:attribute:: scale_cbf


   .. py:attribute:: mlp_sbf


   .. py:attribute:: scale_sbf_sum


   .. py:attribute:: down_projection


   .. py:attribute:: up_projection_ca


   .. py:attribute:: inv_sqrt_2


   .. py:method:: forward(m, bases, idx, id_swap)

      :returns: **m** -- Edge embeddings (c->a).
      :rtype: torch.Tensor, shape=(nEdges, emb_size_edge)



.. py:class:: TripletInteraction(emb_size_in: int, emb_size_out: int, emb_size_trip_in: int, emb_size_trip_out: int, emb_size_rbf: int, emb_size_cbf: int, symmetric_mp: bool = True, swap_output: bool = True, activation=None)

   Bases: :py:obj:`torch.nn.Module`


   Triplet-based message passing block.

   :param emb_size_in: Embedding size of the input embeddings.
   :type emb_size_in: int
   :param emb_size_out: Embedding size of the output embeddings.
   :type emb_size_out: int
   :param emb_size_trip_in: (Down-projected) embedding size of the quadruplet edge embeddings
                            before the bilinear layer.
   :type emb_size_trip_in: int
   :param emb_size_trip_out: (Down-projected) embedding size of the quadruplet edge embeddings
                             after the bilinear layer.
   :type emb_size_trip_out: int
   :param emb_size_rbf: Embedding size of the radial basis transformation.
   :type emb_size_rbf: int
   :param emb_size_cbf: Embedding size of the circular basis transformation (one angle).
   :type emb_size_cbf: int
   :param symmetric_mp: Whether to use symmetric message passing and
                        update the edges in both directions.
   :type symmetric_mp: bool
   :param swap_output: Whether to swap the output embedding directions.
                       Only relevant if symmetric_mp is False.
   :type swap_output: bool
   :param activation: Name of the activation function to use in the dense layers.
   :type activation: str


   .. py:attribute:: symmetric_mp


   .. py:attribute:: swap_output


   .. py:attribute:: dense_ba


   .. py:attribute:: mlp_rbf


   .. py:attribute:: scale_rbf


   .. py:attribute:: mlp_cbf


   .. py:attribute:: scale_cbf_sum


   .. py:attribute:: down_projection


   .. py:attribute:: up_projection_ca


   .. py:attribute:: inv_sqrt_2


   .. py:method:: forward(m, bases, idx, id_swap, expand_idx=None, idx_agg2=None, idx_agg2_inner=None, agg2_out_size=None)

      :returns: **m** -- Edge embeddings.
      :rtype: torch.Tensor, shape=(nEdges, emb_size_edge)



.. py:class:: PairInteraction(emb_size_atom, emb_size_pair_in, emb_size_pair_out, emb_size_rbf, activation=None)

   Bases: :py:obj:`torch.nn.Module`


   Pair-based message passing block.

   :param emb_size_atom: Embedding size of the atoms.
   :type emb_size_atom: int
   :param emb_size_pair_in: Embedding size of the atom pairs before the bilinear layer.
   :type emb_size_pair_in: int
   :param emb_size_pair_out: Embedding size of the atom pairs after the bilinear layer.
   :type emb_size_pair_out: int
   :param emb_size_rbf: Embedding size of the radial basis transformation.
   :type emb_size_rbf: int
   :param activation: Name of the activation function to use in the dense layers.
   :type activation: str


   .. py:attribute:: bilinear


   .. py:attribute:: scale_rbf_sum


   .. py:attribute:: down_projection


   .. py:attribute:: up_projection


   .. py:attribute:: inv_sqrt_2


   .. py:method:: forward(h, rad_basis, edge_index, target_neighbor_idx)

      :returns: **h** -- Atom embeddings.
      :rtype: torch.Tensor, shape=(num_atoms, emb_size_atom)



