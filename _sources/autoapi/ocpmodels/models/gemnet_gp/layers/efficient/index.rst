:py:mod:`ocpmodels.models.gemnet_gp.layers.efficient`
=====================================================

.. py:module:: ocpmodels.models.gemnet_gp.layers.efficient

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.gemnet_gp.layers.efficient.EfficientInteractionDownProjection
   ocpmodels.models.gemnet_gp.layers.efficient.EfficientInteractionBilinear




.. py:class:: EfficientInteractionDownProjection(num_spherical: int, num_radial: int, emb_size_interm: int)


   Bases: :py:obj:`torch.nn.Module`

   Down projection in the efficient reformulation.

   :param emb_size_interm: Intermediate embedding size (down-projection size).
   :type emb_size_interm: int
   :param kernel_initializer: Initializer of the weight matrix.
   :type kernel_initializer: callable

   .. py:method:: reset_parameters() -> None


   .. py:method:: forward(rbf: torch.Tensor, sph: torch.Tensor, id_ca, id_ragged_idx, Kmax: int) -> Tuple[torch.Tensor, torch.Tensor]

      :param rbf:
      :type rbf: torch.Tensor, shape=(1, nEdges, num_radial)
      :param sph:
      :type sph: torch.Tensor, shape=(nEdges, Kmax, num_spherical)
      :param id_ca:
      :param id_ragged_idx:

      :returns: * **rbf_W1** (*torch.Tensor, shape=(nEdges, emb_size_interm, num_spherical)*)
                * **sph** (*torch.Tensor, shape=(nEdges, Kmax, num_spherical)*) -- Kmax = maximum number of neighbors of the edges



.. py:class:: EfficientInteractionBilinear(emb_size: int, emb_size_interm: int, units_out: int)


   Bases: :py:obj:`torch.nn.Module`

   Efficient reformulation of the bilinear layer and subsequent summation.

   :param units_out: Embedding output size of the bilinear layer.
   :type units_out: int
   :param kernel_initializer: Initializer of the weight matrix.
   :type kernel_initializer: callable

   .. py:method:: reset_parameters() -> None


   .. py:method:: forward(basis: Tuple[torch.Tensor, torch.Tensor], m, id_reduce, id_ragged_idx, edge_offset, Kmax: int) -> torch.Tensor

      :param basis:
      :param m:
      :type m: quadruplets: m = m_db , triplets: m = m_ba
      :param id_reduce:
      :param id_ragged_idx:

      :returns: **m_ca** -- Edge embeddings.
      :rtype: torch.Tensor, shape=(nEdges, units_out)



