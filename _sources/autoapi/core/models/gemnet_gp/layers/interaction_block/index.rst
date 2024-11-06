core.models.gemnet_gp.layers.interaction_block
==============================================

.. py:module:: core.models.gemnet_gp.layers.interaction_block

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_gp.layers.interaction_block.InteractionBlockTripletsOnly
   core.models.gemnet_gp.layers.interaction_block.TripletInteraction


Module Contents
---------------

.. py:class:: InteractionBlockTripletsOnly(emb_size_atom: int, emb_size_edge: int, emb_size_trip: int, emb_size_rbf: int, emb_size_cbf: int, emb_size_bil_trip: int, num_before_skip: int, num_after_skip: int, num_concat: int, num_atom: int, activation: str | None = None, name: str = 'Interaction')

   Bases: :py:obj:`torch.nn.Module`


   Interaction block for GemNet-T/dT.

   :param emb_size_atom: Embedding size of the atoms.
   :type emb_size_atom: int
   :param emb_size_edge: Embedding size of the edges.
   :type emb_size_edge: int
   :param emb_size_trip: (Down-projected) Embedding size in the triplet message passing block.
   :type emb_size_trip: int
   :param emb_size_rbf: Embedding size of the radial basis transformation.
   :type emb_size_rbf: int
   :param emb_size_cbf: Embedding size of the circular basis transformation (one angle).
   :type emb_size_cbf: int
   :param emb_size_bil_trip: Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
   :type emb_size_bil_trip: int
   :param num_before_skip: Number of residual blocks before the first skip connection.
   :type num_before_skip: int
   :param num_after_skip: Number of residual blocks after the first skip connection.
   :type num_after_skip: int
   :param num_concat: Number of residual blocks after the concatenation.
   :type num_concat: int
   :param num_atom: Number of residual blocks in the atom embedding blocks.
   :type num_atom: int
   :param activation: Name of the activation function to use in the dense layers except for the final dense layer.
   :type activation: str


   .. py:attribute:: name


   .. py:attribute:: dense_ca


   .. py:attribute:: trip_interaction


   .. py:attribute:: layers_before_skip


   .. py:attribute:: layers_after_skip


   .. py:attribute:: atom_update


   .. py:attribute:: concat_layer


   .. py:attribute:: residual_m


   .. py:attribute:: inv_sqrt_2


   .. py:method:: forward(h: torch.Tensor, m: torch.Tensor, rbf3, cbf3, id3_ragged_idx, id_swap, id3_ba, id3_ca, rbf_h, idx_s, idx_t, edge_offset, Kmax, nAtoms)

      :returns: * **h** (*torch.Tensor, shape=(nEdges, emb_size_atom)*) -- Atom embeddings.
                * **m** (*torch.Tensor, shape=(nEdges, emb_size_edge)*) -- Edge embeddings (c->a).
                * **Node** (*h*)
                * **Edge** (*m, rbf3, id_swap, rbf_h, idx_s, idx_t, cbf3[0], cbf3[1] (dense)*)
                * **Triplet** (*id3_ragged_idx, id3_ba, id3_ca*)



.. py:class:: TripletInteraction(emb_size_edge: int, emb_size_trip: int, emb_size_bilinear: int, emb_size_rbf: int, emb_size_cbf: int, activation: str | None = None, name: str = 'TripletInteraction', **kwargs)

   Bases: :py:obj:`torch.nn.Module`


   Triplet-based message passing block.

   :param emb_size_edge: Embedding size of the edges.
   :type emb_size_edge: int
   :param emb_size_trip: (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
   :type emb_size_trip: int
   :param emb_size_bilinear: Embedding size of the edge embeddings after the bilinear layer.
   :type emb_size_bilinear: int
   :param emb_size_rbf: Embedding size of the radial basis transformation.
   :type emb_size_rbf: int
   :param emb_size_cbf: Embedding size of the circular basis transformation (one angle).
   :type emb_size_cbf: int
   :param activation: Name of the activation function to use in the dense layers except for the final dense layer.
   :type activation: str


   .. py:attribute:: name


   .. py:attribute:: dense_ba


   .. py:attribute:: mlp_rbf


   .. py:attribute:: scale_rbf


   .. py:attribute:: mlp_cbf


   .. py:attribute:: scale_cbf_sum


   .. py:attribute:: down_projection


   .. py:attribute:: up_projection_ca


   .. py:attribute:: up_projection_ac


   .. py:attribute:: inv_sqrt_2


   .. py:method:: forward(m: torch.Tensor, rbf3, cbf3, id3_ragged_idx, id_swap, id3_ba, id3_ca, edge_offset, Kmax)

      :returns: **m** -- Edge embeddings (c->a).
      :rtype: torch.Tensor, shape=(nEdges, emb_size_edge)



