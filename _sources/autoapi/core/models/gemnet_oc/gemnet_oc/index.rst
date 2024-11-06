core.models.gemnet_oc.gemnet_oc
===============================

.. py:module:: core.models.gemnet_oc.gemnet_oc

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.
   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.gemnet_oc.gemnet_oc.GemNetOC
   core.models.gemnet_oc.gemnet_oc.GemNetOCBackbone
   core.models.gemnet_oc.gemnet_oc.GemNetOCEnergyAndGradForceHead
   core.models.gemnet_oc.gemnet_oc.GemNetOCForceHead


Module Contents
---------------

.. py:class:: GemNetOC(num_spherical: int, num_radial: int, num_blocks: int, emb_size_atom: int, emb_size_edge: int, emb_size_trip_in: int, emb_size_trip_out: int, emb_size_quad_in: int, emb_size_quad_out: int, emb_size_aint_in: int, emb_size_aint_out: int, emb_size_rbf: int, emb_size_cbf: int, emb_size_sbf: int, num_before_skip: int, num_after_skip: int, num_concat: int, num_atom: int, num_output_afteratom: int, num_atom_emb_layers: int = 0, num_global_out_layers: int = 2, regress_forces: bool = True, direct_forces: bool = False, use_pbc: bool = True, use_pbc_single: bool = False, scale_backprop_forces: bool = False, cutoff: float = 6.0, cutoff_qint: float | None = None, cutoff_aeaint: float | None = None, cutoff_aint: float | None = None, max_neighbors: int = 50, max_neighbors_qint: int | None = None, max_neighbors_aeaint: int | None = None, max_neighbors_aint: int | None = None, enforce_max_neighbors_strictly: bool = True, rbf: dict[str, str] | None = None, rbf_spherical: dict | None = None, envelope: dict[str, str | int] | None = None, cbf: dict[str, str] | None = None, sbf: dict[str, str] | None = None, extensive: bool = True, forces_coupled: bool = False, output_init: str = 'HeOrthogonal', activation: str = 'silu', quad_interaction: bool = False, atom_edge_interaction: bool = False, edge_atom_interaction: bool = False, atom_interaction: bool = False, scale_basis: bool = False, qint_tags: list | None = None, num_elements: int = 83, otf_graph: bool = False, scale_file: str | None = None, **kwargs)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.GraphModelMixin`


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
   :param emb_size_trip_in: (Down-projected) embedding size of the quadruplet edge embeddings
                            before the bilinear layer.
   :type emb_size_trip_in: int
   :param emb_size_trip_out: (Down-projected) embedding size of the quadruplet edge embeddings
                             after the bilinear layer.
   :type emb_size_trip_out: int
   :param emb_size_quad_in: (Down-projected) embedding size of the quadruplet edge embeddings
                            before the bilinear layer.
   :type emb_size_quad_in: int
   :param emb_size_quad_out: (Down-projected) embedding size of the quadruplet edge embeddings
                             after the bilinear layer.
   :type emb_size_quad_out: int
   :param emb_size_aint_in: Embedding size in the atom interaction before the bilinear layer.
   :type emb_size_aint_in: int
   :param emb_size_aint_out: Embedding size in the atom interaction after the bilinear layer.
   :type emb_size_aint_out: int
   :param emb_size_rbf: Embedding size of the radial basis transformation.
   :type emb_size_rbf: int
   :param emb_size_cbf: Embedding size of the circular basis transformation (one angle).
   :type emb_size_cbf: int
   :param emb_size_sbf: Embedding size of the spherical basis transformation (two angles).
   :type emb_size_sbf: int
   :param num_before_skip: Number of residual blocks before the first skip connection.
   :type num_before_skip: int
   :param num_after_skip: Number of residual blocks after the first skip connection.
   :type num_after_skip: int
   :param num_concat: Number of residual blocks after the concatenation.
   :type num_concat: int
   :param num_atom: Number of residual blocks in the atom embedding blocks.
   :type num_atom: int
   :param num_output_afteratom: Number of residual blocks in the output blocks
                                after adding the atom embedding.
   :type num_output_afteratom: int
   :param num_atom_emb_layers: Number of residual blocks for transforming atom embeddings.
   :type num_atom_emb_layers: int
   :param num_global_out_layers: Number of final residual blocks before the output.
   :type num_global_out_layers: int
   :param regress_forces: Whether to predict forces. Default: True
   :type regress_forces: bool
   :param direct_forces: If True predict forces based on aggregation of interatomic directions.
                         If False predict forces based on negative gradient of energy potential.
   :type direct_forces: bool
   :param use_pbc: Whether to use periodic boundary conditions.
   :type use_pbc: bool
   :param use_pbc_single: Process batch PBC graphs one at a time
   :param scale_backprop_forces: Whether to scale up the energy and then scales down the forces
                                 to prevent NaNs and infs in backpropagated forces.
   :type scale_backprop_forces: bool
   :param cutoff: Embedding cutoff for interatomic connections and embeddings in Angstrom.
   :type cutoff: float
   :param cutoff_qint: Quadruplet interaction cutoff in Angstrom.
                       Optional. Uses cutoff per default.
   :type cutoff_qint: float
   :param cutoff_aeaint: Edge-to-atom and atom-to-edge interaction cutoff in Angstrom.
                         Optional. Uses cutoff per default.
   :type cutoff_aeaint: float
   :param cutoff_aint: Atom-to-atom interaction cutoff in Angstrom.
                       Optional. Uses maximum of all other cutoffs per default.
   :type cutoff_aint: float
   :param max_neighbors: Maximum number of neighbors for interatomic connections and embeddings.
   :type max_neighbors: int
   :param max_neighbors_qint: Maximum number of quadruplet interactions per embedding.
                              Optional. Uses max_neighbors per default.
   :type max_neighbors_qint: int
   :param max_neighbors_aeaint: Maximum number of edge-to-atom and atom-to-edge interactions per embedding.
                                Optional. Uses max_neighbors per default.
   :type max_neighbors_aeaint: int
   :param max_neighbors_aint: Maximum number of atom-to-atom interactions per atom.
                              Optional. Uses maximum of all other neighbors per default.
   :type max_neighbors_aint: int
   :param enforce_max_neighbors_strictly: When subselected edges based on max_neighbors args, arbitrarily
                                          select amongst degenerate edges to have exactly the correct number.
   :type enforce_max_neighbors_strictly: bool
   :param rbf: Name and hyperparameters of the radial basis function.
   :type rbf: dict
   :param rbf_spherical: Name and hyperparameters of the radial basis function used as part of the
                         circular and spherical bases.
                         Optional. Uses rbf per default.
   :type rbf_spherical: dict
   :param envelope: Name and hyperparameters of the envelope function.
   :type envelope: dict
   :param cbf: Name and hyperparameters of the circular basis function.
   :type cbf: dict
   :param sbf: Name and hyperparameters of the spherical basis function.
   :type sbf: dict
   :param extensive: Whether the output should be extensive (proportional to the number of atoms)
   :type extensive: bool
   :param forces_coupled: If True, enforce that |F_st| = |F_ts|. No effect if direct_forces is False.
   :type forces_coupled: bool
   :param output_init: Initialization method for the final dense layer.
   :type output_init: str
   :param activation: Name of the activation function.
   :type activation: str
   :param scale_file: Path to the pytorch file containing the scaling factors.
   :type scale_file: str
   :param quad_interaction: Whether to use quadruplet interactions (with dihedral angles)
   :type quad_interaction: bool
   :param atom_edge_interaction: Whether to use atom-to-edge interactions
   :type atom_edge_interaction: bool
   :param edge_atom_interaction: Whether to use edge-to-atom interactions
   :type edge_atom_interaction: bool
   :param atom_interaction: Whether to use atom-to-atom interactions
   :type atom_interaction: bool
   :param scale_basis: Whether to use a scaling layer in the raw basis function for better
                       numerical stability.
   :type scale_basis: bool
   :param qint_tags: Which atom tags to use quadruplet interactions for.
                     0=sub-surface bulk, 1=surface, 2=adsorbate atoms.
   :type qint_tags: list


   .. py:attribute:: num_blocks


   .. py:attribute:: extensive


   .. py:attribute:: activation


   .. py:attribute:: atom_edge_interaction


   .. py:attribute:: edge_atom_interaction


   .. py:attribute:: atom_interaction


   .. py:attribute:: quad_interaction


   .. py:attribute:: qint_tags


   .. py:attribute:: otf_graph


   .. py:attribute:: enforce_max_neighbors_strictly


   .. py:attribute:: use_pbc


   .. py:attribute:: use_pbc_single


   .. py:attribute:: direct_forces


   .. py:attribute:: forces_coupled


   .. py:attribute:: regress_forces


   .. py:attribute:: force_scaler


   .. py:attribute:: atom_emb


   .. py:attribute:: edge_emb


   .. py:attribute:: int_blocks


   .. py:attribute:: out_blocks


   .. py:attribute:: out_mlp_E


   .. py:attribute:: out_energy


   .. py:method:: set_cutoffs(cutoff, cutoff_qint, cutoff_aeaint, cutoff_aint)


   .. py:method:: set_max_neighbors(max_neighbors, max_neighbors_qint, max_neighbors_aeaint, max_neighbors_aint)


   .. py:method:: init_basis_functions(num_radial, num_spherical, rbf, rbf_spherical, envelope, cbf, sbf, scale_basis)


   .. py:method:: init_shared_basis_layers(num_radial, num_spherical, emb_size_rbf, emb_size_cbf, emb_size_sbf)


   .. py:method:: calculate_quad_angles(V_st, V_qint_st, quad_idx)

      Calculate angles for quadruplet-based message passing.

      :param V_st: Normalized directions from s to t
      :type V_st: Tensor, shape = (nAtoms, 3)
      :param V_qint_st: Normalized directions from s to t for the quadruplet
                        interaction graph
      :type V_qint_st: Tensor, shape = (nAtoms, 3)
      :param quad_idx: Indices relevant for quadruplet interactions.
      :type quad_idx: dict of torch.Tensor

      :returns: * **cosφ_cab** (*Tensor, shape = (num_triplets_inint,)*) -- Cosine of angle between atoms c -> a <- b.
                * **cosφ_abd** (*Tensor, shape = (num_triplets_qint,)*) -- Cosine of angle between atoms a -> b -> d.
                * **angle_cabd** (*Tensor, shape = (num_quadruplets,)*) -- Dihedral angle between atoms c <- a-b -> d.



   .. py:method:: select_symmetric_edges(tensor: torch.Tensor, mask: torch.Tensor, reorder_idx: torch.Tensor, opposite_neg) -> torch.Tensor

      Use a mask to remove values of removed edges and then
      duplicate the values for the correct edge direction.

      :param tensor: Values to symmetrize for the new tensor.
      :type tensor: torch.Tensor
      :param mask: Mask defining which edges go in the correct direction.
      :type mask: torch.Tensor
      :param reorder_idx: Indices defining how to reorder the tensor values after
                          concatenating the edge values of both directions.
      :type reorder_idx: torch.Tensor
      :param opposite_neg: Whether the edge in the opposite direction should use the
                           negative tensor value.
      :type opposite_neg: bool

      :returns: **tensor_ordered** -- A tensor with symmetrized values.
      :rtype: torch.Tensor



   .. py:method:: symmetrize_edges(graph, batch_idx)

      Symmetrize edges to ensure existence of counter-directional edges.

      Some edges are only present in one direction in the data,
      since every atom has a maximum number of neighbors.
      We only use i->j edges here. So we lose some j->i edges
      and add others by making it symmetric.



   .. py:method:: subselect_edges(data, graph, cutoff=None, max_neighbors=None)

      Subselect edges using a stricter cutoff and max_neighbors.



   .. py:method:: generate_graph_dict(data, cutoff, max_neighbors)

      Generate a radius/nearest neighbor graph.



   .. py:method:: subselect_graph(data, graph, cutoff, max_neighbors, cutoff_orig, max_neighbors_orig)

      If the new cutoff and max_neighbors is different from the original,
      subselect the edges of a given graph.



   .. py:method:: get_graphs_and_indices(data)

      "Generate embedding and interaction graphs and indices.



   .. py:method:: get_bases(main_graph, a2a_graph, a2ee2a_graph, qint_graph, trip_idx_e2e, trip_idx_a2e, trip_idx_e2a, quad_idx, num_atoms)

      Calculate and transform basis functions.



   .. py:method:: forward(data)


   .. py:property:: num_params
      :type: int



.. py:class:: GemNetOCBackbone(num_spherical: int, num_radial: int, num_blocks: int, emb_size_atom: int, emb_size_edge: int, emb_size_trip_in: int, emb_size_trip_out: int, emb_size_quad_in: int, emb_size_quad_out: int, emb_size_aint_in: int, emb_size_aint_out: int, emb_size_rbf: int, emb_size_cbf: int, emb_size_sbf: int, num_before_skip: int, num_after_skip: int, num_concat: int, num_atom: int, num_output_afteratom: int, num_atom_emb_layers: int = 0, num_global_out_layers: int = 2, regress_forces: bool = True, direct_forces: bool = False, use_pbc: bool = True, use_pbc_single: bool = False, scale_backprop_forces: bool = False, cutoff: float = 6.0, cutoff_qint: float | None = None, cutoff_aeaint: float | None = None, cutoff_aint: float | None = None, max_neighbors: int = 50, max_neighbors_qint: int | None = None, max_neighbors_aeaint: int | None = None, max_neighbors_aint: int | None = None, enforce_max_neighbors_strictly: bool = True, rbf: dict[str, str] | None = None, rbf_spherical: dict | None = None, envelope: dict[str, str | int] | None = None, cbf: dict[str, str] | None = None, sbf: dict[str, str] | None = None, extensive: bool = True, forces_coupled: bool = False, output_init: str = 'HeOrthogonal', activation: str = 'silu', quad_interaction: bool = False, atom_edge_interaction: bool = False, edge_atom_interaction: bool = False, atom_interaction: bool = False, scale_basis: bool = False, qint_tags: list | None = None, num_elements: int = 83, otf_graph: bool = False, scale_file: str | None = None, **kwargs)

   Bases: :py:obj:`GemNetOC`, :py:obj:`fairchem.core.models.base.BackboneInterface`


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
   :param emb_size_trip_in: (Down-projected) embedding size of the quadruplet edge embeddings
                            before the bilinear layer.
   :type emb_size_trip_in: int
   :param emb_size_trip_out: (Down-projected) embedding size of the quadruplet edge embeddings
                             after the bilinear layer.
   :type emb_size_trip_out: int
   :param emb_size_quad_in: (Down-projected) embedding size of the quadruplet edge embeddings
                            before the bilinear layer.
   :type emb_size_quad_in: int
   :param emb_size_quad_out: (Down-projected) embedding size of the quadruplet edge embeddings
                             after the bilinear layer.
   :type emb_size_quad_out: int
   :param emb_size_aint_in: Embedding size in the atom interaction before the bilinear layer.
   :type emb_size_aint_in: int
   :param emb_size_aint_out: Embedding size in the atom interaction after the bilinear layer.
   :type emb_size_aint_out: int
   :param emb_size_rbf: Embedding size of the radial basis transformation.
   :type emb_size_rbf: int
   :param emb_size_cbf: Embedding size of the circular basis transformation (one angle).
   :type emb_size_cbf: int
   :param emb_size_sbf: Embedding size of the spherical basis transformation (two angles).
   :type emb_size_sbf: int
   :param num_before_skip: Number of residual blocks before the first skip connection.
   :type num_before_skip: int
   :param num_after_skip: Number of residual blocks after the first skip connection.
   :type num_after_skip: int
   :param num_concat: Number of residual blocks after the concatenation.
   :type num_concat: int
   :param num_atom: Number of residual blocks in the atom embedding blocks.
   :type num_atom: int
   :param num_output_afteratom: Number of residual blocks in the output blocks
                                after adding the atom embedding.
   :type num_output_afteratom: int
   :param num_atom_emb_layers: Number of residual blocks for transforming atom embeddings.
   :type num_atom_emb_layers: int
   :param num_global_out_layers: Number of final residual blocks before the output.
   :type num_global_out_layers: int
   :param regress_forces: Whether to predict forces. Default: True
   :type regress_forces: bool
   :param direct_forces: If True predict forces based on aggregation of interatomic directions.
                         If False predict forces based on negative gradient of energy potential.
   :type direct_forces: bool
   :param use_pbc: Whether to use periodic boundary conditions.
   :type use_pbc: bool
   :param use_pbc_single: Process batch PBC graphs one at a time
   :param scale_backprop_forces: Whether to scale up the energy and then scales down the forces
                                 to prevent NaNs and infs in backpropagated forces.
   :type scale_backprop_forces: bool
   :param cutoff: Embedding cutoff for interatomic connections and embeddings in Angstrom.
   :type cutoff: float
   :param cutoff_qint: Quadruplet interaction cutoff in Angstrom.
                       Optional. Uses cutoff per default.
   :type cutoff_qint: float
   :param cutoff_aeaint: Edge-to-atom and atom-to-edge interaction cutoff in Angstrom.
                         Optional. Uses cutoff per default.
   :type cutoff_aeaint: float
   :param cutoff_aint: Atom-to-atom interaction cutoff in Angstrom.
                       Optional. Uses maximum of all other cutoffs per default.
   :type cutoff_aint: float
   :param max_neighbors: Maximum number of neighbors for interatomic connections and embeddings.
   :type max_neighbors: int
   :param max_neighbors_qint: Maximum number of quadruplet interactions per embedding.
                              Optional. Uses max_neighbors per default.
   :type max_neighbors_qint: int
   :param max_neighbors_aeaint: Maximum number of edge-to-atom and atom-to-edge interactions per embedding.
                                Optional. Uses max_neighbors per default.
   :type max_neighbors_aeaint: int
   :param max_neighbors_aint: Maximum number of atom-to-atom interactions per atom.
                              Optional. Uses maximum of all other neighbors per default.
   :type max_neighbors_aint: int
   :param enforce_max_neighbors_strictly: When subselected edges based on max_neighbors args, arbitrarily
                                          select amongst degenerate edges to have exactly the correct number.
   :type enforce_max_neighbors_strictly: bool
   :param rbf: Name and hyperparameters of the radial basis function.
   :type rbf: dict
   :param rbf_spherical: Name and hyperparameters of the radial basis function used as part of the
                         circular and spherical bases.
                         Optional. Uses rbf per default.
   :type rbf_spherical: dict
   :param envelope: Name and hyperparameters of the envelope function.
   :type envelope: dict
   :param cbf: Name and hyperparameters of the circular basis function.
   :type cbf: dict
   :param sbf: Name and hyperparameters of the spherical basis function.
   :type sbf: dict
   :param extensive: Whether the output should be extensive (proportional to the number of atoms)
   :type extensive: bool
   :param forces_coupled: If True, enforce that |F_st| = |F_ts|. No effect if direct_forces is False.
   :type forces_coupled: bool
   :param output_init: Initialization method for the final dense layer.
   :type output_init: str
   :param activation: Name of the activation function.
   :type activation: str
   :param scale_file: Path to the pytorch file containing the scaling factors.
   :type scale_file: str
   :param quad_interaction: Whether to use quadruplet interactions (with dihedral angles)
   :type quad_interaction: bool
   :param atom_edge_interaction: Whether to use atom-to-edge interactions
   :type atom_edge_interaction: bool
   :param edge_atom_interaction: Whether to use edge-to-atom interactions
   :type edge_atom_interaction: bool
   :param atom_interaction: Whether to use atom-to-atom interactions
   :type atom_interaction: bool
   :param scale_basis: Whether to use a scaling layer in the raw basis function for better
                       numerical stability.
   :type scale_basis: bool
   :param qint_tags: Which atom tags to use quadruplet interactions for.
                     0=sub-surface bulk, 1=surface, 2=adsorbate atoms.
   :type qint_tags: list


   .. py:method:: forward(data: torch_geometric.data.batch.Batch) -> dict[str, torch.Tensor]

      Backbone forward.

      :param data: Atomic systems as input
      :type data: DataBatch

      :returns: **embedding** -- Return backbone embeddings for the given input
      :rtype: dict[str->torch.Tensor]



.. py:class:: GemNetOCEnergyAndGradForceHead(backbone: fairchem.core.models.base.BackboneInterface, num_global_out_layers: int, output_init: str = 'HeOrthogonal')

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


   .. py:attribute:: force_scaler


   .. py:attribute:: out_mlp_E


   .. py:attribute:: out_energy


   .. py:method:: forward(data: torch_geometric.data.batch.Batch, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]

      Head forward.

      :param data: Atomic systems as input
      :type data: DataBatch
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



.. py:class:: GemNetOCForceHead(backbone, num_global_out_layers: int, output_init: str = 'HeOrthogonal')

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


   .. py:attribute:: forces_coupled


   .. py:method:: forward(data: torch_geometric.data.batch.Batch, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]

      Head forward.

      :param data: Atomic systems as input
      :type data: DataBatch
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



