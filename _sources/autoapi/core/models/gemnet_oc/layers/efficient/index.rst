core.models.gemnet_oc.layers.efficient
======================================

.. py:module:: core.models.gemnet_oc.layers.efficient

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_oc.layers.efficient.BasisEmbedding
   core.models.gemnet_oc.layers.efficient.EfficientInteractionBilinear


Module Contents
---------------

.. py:class:: BasisEmbedding(num_radial: int, emb_size_interm: int, num_spherical: int | None = None)

   Bases: :py:obj:`torch.nn.Module`


   Embed a basis (CBF, SBF), optionally using the efficient reformulation.

   :param num_radial: Number of radial basis functions.
   :type num_radial: int
   :param emb_size_interm: Intermediate embedding size of triplets/quadruplets.
   :type emb_size_interm: int
   :param num_spherical: Number of circular/spherical basis functions.
                         Only required if there is a circular/spherical basis.
   :type num_spherical: int


   .. py:attribute:: weight
      :type:  torch.nn.Parameter


   .. py:attribute:: num_radial


   .. py:attribute:: num_spherical


   .. py:method:: reset_parameters() -> None


   .. py:method:: forward(rad_basis, sph_basis=None, idx_rad_outer=None, idx_rad_inner=None, idx_sph_outer=None, idx_sph_inner=None, num_atoms=None)

      :param rad_basis: Raw radial basis.
      :type rad_basis: torch.Tensor, shape=(num_edges, num_radial or num_orders * num_radial)
      :param sph_basis: Raw spherical or circular basis.
      :type sph_basis: torch.Tensor, shape=(num_triplets or num_quadruplets, num_spherical)
      :param idx_rad_outer: Atom associated with each radial basis value.
                            Optional, used for efficient edge aggregation.
      :type idx_rad_outer: torch.Tensor, shape=(num_edges)
      :param idx_rad_inner: Enumerates radial basis values per atom.
                            Optional, used for efficient edge aggregation.
      :type idx_rad_inner: torch.Tensor, shape=(num_edges)
      :param idx_sph_outer: Edge associated with each circular/spherical basis value.
                            Optional, used for efficient triplet/quadruplet aggregation.
      :type idx_sph_outer: torch.Tensor, shape=(num_triplets or num_quadruplets)
      :param idx_sph_inner: Enumerates circular/spherical basis values per edge.
                            Optional, used for efficient triplet/quadruplet aggregation.
      :type idx_sph_inner: torch.Tensor, shape=(num_triplets or num_quadruplets)
      :param num_atoms: Total number of atoms.
                        Optional, used for efficient edge aggregation.
      :type num_atoms: int

      :returns: * **rad_W1** (*torch.Tensor, shape=(num_edges, emb_size_interm, num_spherical)*)
                * **sph** (*torch.Tensor, shape=(num_edges, Kmax, num_spherical)*) -- Kmax = maximum number of neighbors of the edges



.. py:class:: EfficientInteractionBilinear(emb_size_in: int, emb_size_interm: int, emb_size_out: int)

   Bases: :py:obj:`torch.nn.Module`


   Efficient reformulation of the bilinear layer and subsequent summation.

   :param emb_size_in: Embedding size of input triplets/quadruplets.
   :type emb_size_in: int
   :param emb_size_interm: Intermediate embedding size of the basis transformation.
   :type emb_size_interm: int
   :param emb_size_out: Embedding size of output triplets/quadruplets.
   :type emb_size_out: int


   .. py:attribute:: emb_size_in


   .. py:attribute:: emb_size_interm


   .. py:attribute:: emb_size_out


   .. py:attribute:: bilinear


   .. py:method:: forward(basis, m, idx_agg_outer, idx_agg_inner, idx_agg2_outer=None, idx_agg2_inner=None, agg2_out_size=None)

      :param basis:
                    shapes=((num_edges, emb_size_interm, num_spherical),
                            (num_edges, num_spherical, Kmax))
                    First element: Radial basis multiplied with weight matrix
                    Second element: Circular/spherical basis
      :type basis: Tuple (torch.Tensor, torch.Tensor),
      :param m: Input edge embeddings
      :type m: torch.Tensor, shape=(num_edges, emb_size_in)
      :param idx_agg_outer: Output edge aggregating this intermediate triplet/quadruplet edge.
      :type idx_agg_outer: torch.Tensor, shape=(num_triplets or num_quadruplets)
      :param idx_agg_inner: Enumerates intermediate edges per output edge.
      :type idx_agg_inner: torch.Tensor, shape=(num_triplets or num_quadruplets)
      :param idx_agg2_outer: Output atom aggregating this edge.
      :type idx_agg2_outer: torch.Tensor, shape=(num_edges)
      :param idx_agg2_inner: Enumerates edges per output atom.
      :type idx_agg2_inner: torch.Tensor, shape=(num_edges)
      :param agg2_out_size: Number of output embeddings when aggregating twice. Typically
                            the number of atoms.
      :type agg2_out_size: int

      :returns: **m_ca** -- Aggregated edge/atom embeddings.
      :rtype: torch.Tensor, shape=(num_edges, emb_size)



