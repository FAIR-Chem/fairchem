core.models.gemnet.layers.efficient
===================================

.. py:module:: core.models.gemnet.layers.efficient

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet.layers.efficient.EfficientInteractionDownProjection
   core.models.gemnet.layers.efficient.EfficientInteractionBilinear


Module Contents
---------------

.. py:class:: EfficientInteractionDownProjection(num_spherical: int, num_radial: int, emb_size_interm: int)

   Bases: :py:obj:`torch.nn.Module`


   Down projection in the efficient reformulation.

   :param emb_size_interm: Intermediate embedding size (down-projection size).
   :type emb_size_interm: int
   :param kernel_initializer: Initializer of the weight matrix.
   :type kernel_initializer: callable


   .. py:method:: reset_parameters() -> None


   .. py:method:: forward(rbf, sph, id_ca, id_ragged_idx)

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


   .. py:method:: forward(basis, m, id_reduce, id_ragged_idx) -> torch.Tensor

      :param basis:
      :param m:
      :type m: quadruplets: m = m_db , triplets: m = m_ba
      :param id_reduce:
      :param id_ragged_idx:

      :returns: **m_ca** -- Edge embeddings.
      :rtype: torch.Tensor, shape=(nEdges, units_out)



