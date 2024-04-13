:py:mod:`ocpmodels.models.gemnet_oc.layers.atom_update_block`
=============================================================

.. py:module:: ocpmodels.models.gemnet_oc.layers.atom_update_block

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.gemnet_oc.layers.atom_update_block.AtomUpdateBlock
   ocpmodels.models.gemnet_oc.layers.atom_update_block.OutputBlock




.. py:class:: AtomUpdateBlock(emb_size_atom: int, emb_size_edge: int, emb_size_rbf: int, nHidden: int, activation=None)


   Bases: :py:obj:`torch.nn.Module`

   Aggregate the message embeddings of the atoms

   :param emb_size_atom: Embedding size of the atoms.
   :type emb_size_atom: int
   :param emb_size_edge: Embedding size of the edges.
   :type emb_size_edge: int
   :param emb_size_rbf: Embedding size of the radial basis.
   :type emb_size_rbf: int
   :param nHidden: Number of residual blocks.
   :type nHidden: int
   :param activation: Name of the activation function to use in the dense layers.
   :type activation: callable/str

   .. py:method:: get_mlp(units_in: int, units: int, nHidden: int, activation)


   .. py:method:: forward(h: torch.Tensor, m, basis_rad, idx_atom)

      :returns: **h** -- Atom embedding.
      :rtype: torch.Tensor, shape=(nAtoms, emb_size_atom)



.. py:class:: OutputBlock(emb_size_atom: int, emb_size_edge: int, emb_size_rbf: int, nHidden: int, nHidden_afteratom: int, activation: str | None = None, direct_forces: bool = True)


   Bases: :py:obj:`AtomUpdateBlock`

   Combines the atom update block and subsequent final dense layer.

   :param emb_size_atom: Embedding size of the atoms.
   :type emb_size_atom: int
   :param emb_size_edge: Embedding size of the edges.
   :type emb_size_edge: int
   :param emb_size_rbf: Embedding size of the radial basis.
   :type emb_size_rbf: int
   :param nHidden: Number of residual blocks before adding the atom embedding.
   :type nHidden: int
   :param nHidden_afteratom: Number of residual blocks after adding the atom embedding.
   :type nHidden_afteratom: int
   :param activation: Name of the activation function to use in the dense layers.
   :type activation: str
   :param direct_forces: If true directly predict forces, i.e. without taking the gradient
                         of the energy potential.
   :type direct_forces: bool

   .. py:method:: forward(h: torch.Tensor, m: torch.Tensor, basis_rad, idx_atom)

      :returns: * *torch.Tensor, shape=(nAtoms, emb_size_atom)* -- Output atom embeddings.
                * *torch.Tensor, shape=(nEdges, emb_size_edge)* -- Output edge embeddings.



