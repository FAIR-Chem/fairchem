core.models.gemnet_gp.layers.atom_update_block
==============================================

.. py:module:: core.models.gemnet_gp.layers.atom_update_block

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_gp.layers.atom_update_block.AtomUpdateBlock
   core.models.gemnet_gp.layers.atom_update_block.OutputBlock


Functions
---------

.. autoapisummary::

   core.models.gemnet_gp.layers.atom_update_block.scatter_sum


Module Contents
---------------

.. py:function:: scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1, out: torch.Tensor | None = None, dim_size: int | None = None) -> torch.Tensor

   Clone of torch_scatter.scatter_sum but without in-place operations


.. py:class:: AtomUpdateBlock(emb_size_atom: int, emb_size_edge: int, emb_size_rbf: int, nHidden: int, activation: str | None = None, name: str = 'atom_update')

   Bases: :py:obj:`torch.nn.Module`


   Aggregate the message embeddings of the atoms

   :param emb_size_atom: Embedding size of the atoms.
   :type emb_size_atom: int
   :param emb_size_atom: Embedding size of the edges.
   :type emb_size_atom: int
   :param nHidden: Number of residual blocks.
   :type nHidden: int
   :param activation: Name of the activation function to use in the dense layers.
   :type activation: callable/str


   .. py:method:: get_mlp(units_in: int, units: int, nHidden: int, activation: str | None)


   .. py:method:: forward(nAtoms: int, m: int, rbf, id_j)

      :returns: **h** -- Atom embedding.
      :rtype: torch.Tensor, shape=(nAtoms, emb_size_atom)



.. py:class:: OutputBlock(emb_size_atom: int, emb_size_edge: int, emb_size_rbf: int, nHidden: int, num_targets: int, activation: str | None = None, direct_forces: bool = True, output_init: str = 'HeOrthogonal', name: str = 'output', **kwargs)

   Bases: :py:obj:`AtomUpdateBlock`


   Combines the atom update block and subsequent final dense layer.

   :param emb_size_atom: Embedding size of the atoms.
   :type emb_size_atom: int
   :param emb_size_atom: Embedding size of the edges.
   :type emb_size_atom: int
   :param nHidden: Number of residual blocks.
   :type nHidden: int
   :param num_targets: Number of targets.
   :type num_targets: int
   :param activation: Name of the activation function to use in the dense layers except for the final dense layer.
   :type activation: str
   :param direct_forces: If true directly predict forces without taking the gradient of the energy potential.
   :type direct_forces: bool
   :param output_init: Kernel initializer of the final dense layer.
   :type output_init: int


   .. py:attribute:: dense_rbf_F
      :type:  core.models.gemnet_gp.layers.base_layers.Dense


   .. py:attribute:: out_forces
      :type:  core.models.gemnet_gp.layers.base_layers.Dense


   .. py:attribute:: out_energy
      :type:  core.models.gemnet_gp.layers.base_layers.Dense


   .. py:method:: reset_parameters() -> None


   .. py:method:: forward(nAtoms: int, m, rbf, id_j: torch.Tensor)

      :returns: * **(E, F)** (*tuple*)
                * **- E** (*torch.Tensor, shape=(nAtoms, num_targets)*)
                * **- F** (*torch.Tensor, shape=(nEdges, num_targets)*)
                * *Energy and force prediction*



