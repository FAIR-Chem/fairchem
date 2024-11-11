core.models.equiformer_v2.heads.rank2
=====================================

.. py:module:: core.models.equiformer_v2.heads.rank2

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.equiformer_v2.heads.rank2.Rank2Block
   core.models.equiformer_v2.heads.rank2.Rank2DecompositionEdgeBlock
   core.models.equiformer_v2.heads.rank2.Rank2SymmetricTensorHead


Module Contents
---------------

.. py:class:: Rank2Block(emb_size: int, num_layers: int = 2, edge_level: bool = False, extensive: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   Output block for predicting rank-2 tensors (stress, dielectric tensor).
   Applies outer product between edges and computes node-wise or edge-wise MLP.

   :param emb_size: Size of edge embedding used to compute outer product
   :type emb_size: int
   :param num_layers: Number of layers of the MLP
   :type num_layers: int
   :param edge_level: If true apply MLP at edge level before pooling, otherwise use MLP at nodes after pooling
   :type edge_level: bool
   :param extensive: Whether to sum or average the outer products
   :type extensive: bool


   .. py:attribute:: edge_level


   .. py:attribute:: emb_size


   .. py:attribute:: extensive


   .. py:attribute:: scalar_nonlinearity


   .. py:attribute:: r2tensor_MLP


   .. py:method:: forward(edge_distance_vec, x_edge, edge_index, data)

      :param edge_distance_vec: Tensor of shape (..., 3)
      :type edge_distance_vec: torch.Tensor
      :param x_edge: Tensor of shape (..., emb_size)
      :type x_edge: torch.Tensor
      :param edge_index: Tensor of shape (2, nEdges)
      :type edge_index: torch.Tensor
      :param data: LMDBDataset sample



.. py:class:: Rank2DecompositionEdgeBlock(emb_size: int, num_layers: int = 2, edge_level: bool = False, extensive: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   Output block for predicting rank-2 tensors (stress, dielectric tensor, etc).
   Decomposes a rank-2 symmetric tensor into irrep degree 0 and 2.

   :param emb_size: Size of edge embedding used to compute outer product
   :type emb_size: int
   :param num_layers: Number of layers of the MLP
   :type num_layers: int
   :param edge_level: If true apply MLP at edge level before pooling, otherwise use MLP at nodes after pooling
   :type edge_level: bool
   :param extensive: Whether to sum or average the outer products
   :type extensive: bool


   .. py:attribute:: emb_size


   .. py:attribute:: edge_level


   .. py:attribute:: extensive


   .. py:attribute:: scalar_nonlinearity


   .. py:attribute:: scalar_MLP


   .. py:attribute:: irrep2_MLP


   .. py:attribute:: change_mat


   .. py:method:: forward(edge_distance_vec, x_edge, edge_index, data)

      :param edge_distance_vec: Tensor of shape (..., 3)
      :type edge_distance_vec: torch.Tensor
      :param x_edge: Tensor of shape (..., emb_size)
      :type x_edge: torch.Tensor
      :param edge_index: Tensor of shape (2, nEdges)
      :type edge_index: torch.Tensor
      :param data: LMDBDataset sample



.. py:class:: Rank2SymmetricTensorHead(backbone: fairchem.core.models.base.BackboneInterface, output_name: str = 'stress', decompose: bool = False, edge_level_mlp: bool = False, num_mlp_layers: int = 2, use_source_target_embedding: bool = False, extensive: bool = False, avg_num_nodes: int = 1.0, default_norm_type: str = 'layer_norm_sh')

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.HeadInterface`


   A rank 2 symmetric tensor prediction head.

   .. attribute:: ouput_name

      name of output prediction property (ie, stress)

   .. attribute:: sphharm_norm

      layer normalization for spherical harmonic edge weights

   .. attribute:: xedge_layer_norm

      embedding layer norm

   .. attribute:: block

      rank 2 equivariant symmetric tensor block


   .. py:attribute:: output_name


   .. py:attribute:: decompose


   .. py:attribute:: use_source_target_embedding


   .. py:attribute:: avg_num_nodes


   .. py:attribute:: sphharm_norm


   .. py:attribute:: xedge_layer_norm


   .. py:method:: forward(data: dict[str, torch.Tensor] | torch.Tensor, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]

      :param data: data batch
      :param emb: dictionary with embedding object and graph data

      Returns: dict of {output property name: predicted value}



