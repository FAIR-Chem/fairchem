core.models.gemnet.gemnet
=========================

.. py:module:: core.models.gemnet.gemnet

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet.gemnet.GemNetT
   core.models.gemnet.gemnet.GemNetTBackbone
   core.models.gemnet.gemnet.GemNetTEnergyAndGradForceHead
   core.models.gemnet.gemnet.GemNetTForceHead


Module Contents
---------------

.. py:class:: GemNetT(num_spherical: int, num_radial: int, num_blocks: int, emb_size_atom: int, emb_size_edge: int, emb_size_trip: int, emb_size_rbf: int, emb_size_cbf: int, emb_size_bil_trip: int, num_before_skip: int, num_after_skip: int, num_concat: int, num_atom: int, regress_forces: bool = True, direct_forces: bool = False, cutoff: float = 6.0, max_neighbors: int = 50, rbf: dict | None = None, envelope: dict | None = None, cbf: dict | None = None, extensive: bool = True, otf_graph: bool = False, use_pbc: bool = True, use_pbc_single: bool = False, output_init: str = 'HeOrthogonal', activation: str = 'swish', num_elements: int = 83, scale_file: str | None = None)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.GraphModelMixin`


   GemNet-T, triplets-only variant of GemNet

   :param num_spherical: Controls maximum frequency.
   :type num_spherical: int
   :param num_radial: Controls maximum frequency.
   :type num_radial: int
   :param num_blocks: Number of building blocks to be stacked.
   :type num_blocks: int
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
   :param regress_forces: Whether to predict forces. Default: True
   :type regress_forces: bool
   :param direct_forces: If True predict forces based on aggregation of interatomic directions.
                         If False predict forces based on negative gradient of energy potential.
   :type direct_forces: bool
   :param cutoff: Embedding cutoff for interactomic directions in Angstrom.
   :type cutoff: float
   :param rbf: Name and hyperparameters of the radial basis function.
   :type rbf: dict
   :param envelope: Name and hyperparameters of the envelope function.
   :type envelope: dict
   :param cbf: Name and hyperparameters of the cosine basis function.
   :type cbf: dict
   :param extensive: Whether the output should be extensive (proportional to the number of atoms)
   :type extensive: bool
   :param output_init: Initialization method for the final dense layer.
   :type output_init: str
   :param activation: Name of the activation function.
   :type activation: str
   :param scale_file: Path to the json file containing the scaling factors.
   :type scale_file: str


   .. py:attribute:: num_blocks


   .. py:attribute:: extensive


   .. py:attribute:: cutoff


   .. py:attribute:: max_neighbors


   .. py:attribute:: regress_forces


   .. py:attribute:: otf_graph


   .. py:attribute:: use_pbc


   .. py:attribute:: use_pbc_single


   .. py:attribute:: direct_forces


   .. py:attribute:: radial_basis


   .. py:attribute:: cbf_basis3


   .. py:attribute:: mlp_rbf3


   .. py:attribute:: mlp_cbf3


   .. py:attribute:: mlp_rbf_h


   .. py:attribute:: mlp_rbf_out


   .. py:attribute:: atom_emb


   .. py:attribute:: edge_emb


   .. py:attribute:: out_blocks


   .. py:attribute:: int_blocks


   .. py:attribute:: shared_parameters


   .. py:method:: get_triplets(edge_index, num_atoms)

      Get all b->a for each edge c->a.
      It is possible that b=c, as long as the edges are distinct.

      :returns: * **id3_ba** (*torch.Tensor, shape (num_triplets,)*) -- Indices of input edge b->a of each triplet b->a<-c
                * **id3_ca** (*torch.Tensor, shape (num_triplets,)*) -- Indices of output edge c->a of each triplet b->a<-c
                * **id3_ragged_idx** (*torch.Tensor, shape (num_triplets,)*) -- Indices enumerating the copies of id3_ca for creating a padded matrix



   .. py:method:: select_symmetric_edges(tensor: torch.Tensor, mask: torch.Tensor, reorder_idx: torch.Tensor, inverse_neg) -> torch.Tensor


   .. py:method:: reorder_symmetric_edges(edge_index, cell_offsets, neighbors, edge_dist, edge_vector)

      Reorder edges to make finding counter-directional edges easier.

      Some edges are only present in one direction in the data,
      since every atom has a maximum number of neighbors. Since we only use i->j
      edges here, we lose some j->i edges and add others by
      making it symmetric.
      We could fix this by merging edge_index with its counter-edges,
      including the cell_offsets, and then running torch.unique.
      But this does not seem worth it.



   .. py:method:: select_edges(data, edge_index, cell_offsets, neighbors, edge_dist, edge_vector, cutoff=None)


   .. py:method:: generate_interaction_graph(data)


   .. py:method:: forward(data)


   .. py:property:: num_params


.. py:class:: GemNetTBackbone(num_spherical: int, num_radial: int, num_blocks: int, emb_size_atom: int, emb_size_edge: int, emb_size_trip: int, emb_size_rbf: int, emb_size_cbf: int, emb_size_bil_trip: int, num_before_skip: int, num_after_skip: int, num_concat: int, num_atom: int, regress_forces: bool = True, direct_forces: bool = False, cutoff: float = 6.0, max_neighbors: int = 50, rbf: dict | None = None, envelope: dict | None = None, cbf: dict | None = None, extensive: bool = True, otf_graph: bool = False, use_pbc: bool = True, use_pbc_single: bool = False, output_init: str = 'HeOrthogonal', activation: str = 'swish', num_elements: int = 83, scale_file: str | None = None)

   Bases: :py:obj:`GemNetT`, :py:obj:`fairchem.core.models.base.BackboneInterface`


   GemNet-T, triplets-only variant of GemNet

   :param num_spherical: Controls maximum frequency.
   :type num_spherical: int
   :param num_radial: Controls maximum frequency.
   :type num_radial: int
   :param num_blocks: Number of building blocks to be stacked.
   :type num_blocks: int
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
   :param regress_forces: Whether to predict forces. Default: True
   :type regress_forces: bool
   :param direct_forces: If True predict forces based on aggregation of interatomic directions.
                         If False predict forces based on negative gradient of energy potential.
   :type direct_forces: bool
   :param cutoff: Embedding cutoff for interactomic directions in Angstrom.
   :type cutoff: float
   :param rbf: Name and hyperparameters of the radial basis function.
   :type rbf: dict
   :param envelope: Name and hyperparameters of the envelope function.
   :type envelope: dict
   :param cbf: Name and hyperparameters of the cosine basis function.
   :type cbf: dict
   :param extensive: Whether the output should be extensive (proportional to the number of atoms)
   :type extensive: bool
   :param output_init: Initialization method for the final dense layer.
   :type output_init: str
   :param activation: Name of the activation function.
   :type activation: str
   :param scale_file: Path to the json file containing the scaling factors.
   :type scale_file: str


   .. py:method:: forward(data: torch_geometric.data.batch.Batch) -> dict[str, torch.Tensor]

      Backbone forward.

      :param data: Atomic systems as input
      :type data: DataBatch

      :returns: **embedding** -- Return backbone embeddings for the given input
      :rtype: dict[str->torch.Tensor]



.. py:class:: GemNetTEnergyAndGradForceHead(backbone)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.HeadInterface`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: extensive


   .. py:attribute:: regress_forces


   .. py:attribute:: direct_forces


   .. py:method:: forward(data: torch_geometric.data.batch.Batch, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]

      Head forward.

      :param data: Atomic systems as input
      :type data: DataBatch
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



.. py:class:: GemNetTForceHead(backbone)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.HeadInterface`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: direct_forces


   .. py:method:: forward(data: torch_geometric.data.batch.Batch, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]

      Head forward.

      :param data: Atomic systems as input
      :type data: DataBatch
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



