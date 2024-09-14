core.models.equiformer_v2.prediction_heads
==========================================

.. py:module:: core.models.equiformer_v2.prediction_heads


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/models/equiformer_v2/prediction_heads/rank2/index


Classes
-------

.. autoapisummary::

   core.models.equiformer_v2.prediction_heads.Rank2SymmetricTensorHead


Package Contents
----------------

.. py:class:: Rank2SymmetricTensorHead(backbone: fairchem.core.models.base.BackboneInterface, output_name: str, decompose: bool = False, edge_level_mlp: bool = False, num_mlp_layers: int = 2, use_source_target_embedding: bool = False, extensive: bool = False, avg_num_nodes: int = 1.0, default_norm_type: str = 'layer_norm_sh')

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



