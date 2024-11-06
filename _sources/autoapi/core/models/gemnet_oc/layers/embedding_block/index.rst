core.models.gemnet_oc.layers.embedding_block
============================================

.. py:module:: core.models.gemnet_oc.layers.embedding_block

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_oc.layers.embedding_block.AtomEmbedding
   core.models.gemnet_oc.layers.embedding_block.EdgeEmbedding


Module Contents
---------------

.. py:class:: AtomEmbedding(emb_size: int, num_elements: int)

   Bases: :py:obj:`torch.nn.Module`


   Initial atom embeddings based on the atom type

   :param emb_size: Atom embeddings size
   :type emb_size: int


   .. py:attribute:: emb_size


   .. py:attribute:: embeddings


   .. py:method:: forward(Z) -> torch.Tensor

      :returns: **h** -- Atom embeddings.
      :rtype: torch.Tensor, shape=(nAtoms, emb_size)



.. py:class:: EdgeEmbedding(atom_features: int, edge_features: int, out_features: int, activation: str | None = None)

   Bases: :py:obj:`torch.nn.Module`


   Edge embedding based on the concatenation of atom embeddings
   and a subsequent dense layer.

   :param atom_features: Embedding size of the atom embedding.
   :type atom_features: int
   :param edge_features: Embedding size of the input edge embedding.
   :type edge_features: int
   :param out_features: Embedding size after the dense layer.
   :type out_features: int
   :param activation: Activation function used in the dense layer.
   :type activation: str


   .. py:attribute:: dense


   .. py:method:: forward(h: torch.Tensor, m: torch.Tensor, edge_index) -> torch.Tensor

      :param h: Atom embeddings.
      :type h: torch.Tensor, shape (num_atoms, atom_features)
      :param m: Radial basis in embedding block,
                edge embedding in interaction block.
      :type m: torch.Tensor, shape (num_edges, edge_features)

      :returns: **m_st** -- Edge embeddings.
      :rtype: torch.Tensor, shape=(nEdges, emb_size)



