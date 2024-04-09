:py:mod:`ocpmodels.models.gemnet.layers.embedding_block`
========================================================

.. py:module:: ocpmodels.models.gemnet.layers.embedding_block

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.gemnet.layers.embedding_block.AtomEmbedding
   ocpmodels.models.gemnet.layers.embedding_block.EdgeEmbedding




.. py:class:: AtomEmbedding(emb_size, num_elements: int)


   Bases: :py:obj:`torch.nn.Module`

   Initial atom embeddings based on the atom type

   :param emb_size: Atom embeddings size
   :type emb_size: int

   .. py:method:: forward(Z)

      :returns: **h** -- Atom embeddings.
      :rtype: torch.Tensor, shape=(nAtoms, emb_size)



.. py:class:: EdgeEmbedding(atom_features, edge_features, out_features, activation=None)


   Bases: :py:obj:`torch.nn.Module`

   Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

   :param emb_size: Embedding size after the dense layer.
   :type emb_size: int
   :param activation: Activation function used in the dense layer.
   :type activation: str

   .. py:method:: forward(h, m_rbf, idx_s, idx_t)

      :param h:
      :param m_rbf: in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_st
      :type m_rbf: shape (nEdges, nFeatures)
      :param idx_s:
      :param idx_t:

      :returns: **m_st** -- Edge embeddings.
      :rtype: torch.Tensor, shape=(nEdges, emb_size)



