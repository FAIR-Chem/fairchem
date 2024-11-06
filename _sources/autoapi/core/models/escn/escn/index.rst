core.models.escn.escn
=====================

.. py:module:: core.models.escn.escn

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.escn.escn.eSCN
   core.models.escn.escn.eSCNBackbone
   core.models.escn.escn.eSCNEnergyHead
   core.models.escn.escn.eSCNForceHead
   core.models.escn.escn.LayerBlock
   core.models.escn.escn.MessageBlock
   core.models.escn.escn.SO2Block
   core.models.escn.escn.SO2Conv
   core.models.escn.escn.EdgeBlock
   core.models.escn.escn.EnergyBlock
   core.models.escn.escn.ForceBlock


Module Contents
---------------

.. py:class:: eSCN(use_pbc: bool = True, use_pbc_single: bool = False, regress_forces: bool = True, otf_graph: bool = False, max_neighbors: int = 40, cutoff: float = 8.0, max_num_elements: int = 90, num_layers: int = 8, lmax_list: list[int] | None = None, mmax_list: list[int] | None = None, sphere_channels: int = 128, hidden_channels: int = 256, edge_channels: int = 128, num_sphere_samples: int = 128, distance_function: str = 'gaussian', basis_width_scalar: float = 1.0, distance_resolution: float = 0.02, show_timing_info: bool = False, resolution: int | None = None, activation_checkpoint: bool | None = False)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.GraphModelMixin`


   Equivariant Spherical Channel Network
   Paper: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs


   :param use_pbc: Use periodic boundary conditions
   :type use_pbc: bool
   :param use_pbc_single: Process batch PBC graphs one at a time
   :type use_pbc_single: bool
   :param regress_forces: Compute forces
   :type regress_forces: bool
   :param otf_graph: Compute graph On The Fly (OTF)
   :type otf_graph: bool
   :param max_neighbors: Maximum number of neighbors per atom
   :type max_neighbors: int
   :param cutoff: Maximum distance between nieghboring atoms in Angstroms
   :type cutoff: float
   :param max_num_elements: Maximum atomic number
   :type max_num_elements: int
   :param num_layers: Number of layers in the GNN
   :type num_layers: int
   :param lmax_list: List of maximum degree of the spherical harmonics (1 to 10)
   :type lmax_list: int
   :param mmax_list: List of maximum order of the spherical harmonics (0 to lmax)
   :type mmax_list: int
   :param sphere_channels: Number of spherical channels (one set per resolution)
   :type sphere_channels: int
   :param hidden_channels: Number of hidden units in message passing
   :type hidden_channels: int
   :param num_sphere_samples: Number of samples used to approximate the integration of the sphere in the output blocks
   :type num_sphere_samples: int
   :param edge_channels: Number of channels for the edge invariant features
   :type edge_channels: int
   :param distance_function: Basis function used for distances
   :type distance_function: "gaussian", "sigmoid", "linearsigmoid", "silu"
   :param basis_width_scalar: Width of distance basis function
   :type basis_width_scalar: float
   :param distance_resolution: Distance between distance basis functions in Angstroms
   :type distance_resolution: float
   :param show_timing_info: Show timing and memory info
   :type show_timing_info: bool


   .. py:attribute:: activation_checkpoint


   .. py:attribute:: regress_forces


   .. py:attribute:: use_pbc


   .. py:attribute:: use_pbc_single


   .. py:attribute:: cutoff


   .. py:attribute:: otf_graph


   .. py:attribute:: show_timing_info


   .. py:attribute:: max_num_elements


   .. py:attribute:: hidden_channels


   .. py:attribute:: num_layers


   .. py:attribute:: num_atoms
      :value: 0



   .. py:attribute:: num_sphere_samples


   .. py:attribute:: sphere_channels


   .. py:attribute:: max_neighbors


   .. py:attribute:: edge_channels


   .. py:attribute:: distance_resolution


   .. py:attribute:: grad_forces
      :value: False



   .. py:attribute:: lmax_list


   .. py:attribute:: mmax_list


   .. py:attribute:: num_resolutions
      :type:  int


   .. py:attribute:: sphere_channels_all
      :type:  int


   .. py:attribute:: basis_width_scalar


   .. py:attribute:: distance_function


   .. py:attribute:: counter
      :value: 0



   .. py:attribute:: act


   .. py:attribute:: sphere_embedding


   .. py:attribute:: num_gaussians


   .. py:attribute:: SO3_grid


   .. py:attribute:: layer_blocks


   .. py:attribute:: energy_block


   .. py:attribute:: sphere_points


   .. py:attribute:: sphharm_weights


   .. py:method:: forward(data)


   .. py:method:: _init_edge_rot_mat(data, edge_index, edge_distance_vec)


   .. py:property:: num_params
      :type: int



.. py:class:: eSCNBackbone(use_pbc: bool = True, use_pbc_single: bool = False, regress_forces: bool = True, otf_graph: bool = False, max_neighbors: int = 40, cutoff: float = 8.0, max_num_elements: int = 90, num_layers: int = 8, lmax_list: list[int] | None = None, mmax_list: list[int] | None = None, sphere_channels: int = 128, hidden_channels: int = 256, edge_channels: int = 128, num_sphere_samples: int = 128, distance_function: str = 'gaussian', basis_width_scalar: float = 1.0, distance_resolution: float = 0.02, show_timing_info: bool = False, resolution: int | None = None, activation_checkpoint: bool | None = False)

   Bases: :py:obj:`eSCN`, :py:obj:`fairchem.core.models.base.BackboneInterface`


   Equivariant Spherical Channel Network
   Paper: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs


   :param use_pbc: Use periodic boundary conditions
   :type use_pbc: bool
   :param use_pbc_single: Process batch PBC graphs one at a time
   :type use_pbc_single: bool
   :param regress_forces: Compute forces
   :type regress_forces: bool
   :param otf_graph: Compute graph On The Fly (OTF)
   :type otf_graph: bool
   :param max_neighbors: Maximum number of neighbors per atom
   :type max_neighbors: int
   :param cutoff: Maximum distance between nieghboring atoms in Angstroms
   :type cutoff: float
   :param max_num_elements: Maximum atomic number
   :type max_num_elements: int
   :param num_layers: Number of layers in the GNN
   :type num_layers: int
   :param lmax_list: List of maximum degree of the spherical harmonics (1 to 10)
   :type lmax_list: int
   :param mmax_list: List of maximum order of the spherical harmonics (0 to lmax)
   :type mmax_list: int
   :param sphere_channels: Number of spherical channels (one set per resolution)
   :type sphere_channels: int
   :param hidden_channels: Number of hidden units in message passing
   :type hidden_channels: int
   :param num_sphere_samples: Number of samples used to approximate the integration of the sphere in the output blocks
   :type num_sphere_samples: int
   :param edge_channels: Number of channels for the edge invariant features
   :type edge_channels: int
   :param distance_function: Basis function used for distances
   :type distance_function: "gaussian", "sigmoid", "linearsigmoid", "silu"
   :param basis_width_scalar: Width of distance basis function
   :type basis_width_scalar: float
   :param distance_resolution: Distance between distance basis functions in Angstroms
   :type distance_resolution: float
   :param show_timing_info: Show timing and memory info
   :type show_timing_info: bool


   .. py:method:: forward(data: torch_geometric.data.batch.Batch) -> dict[str, torch.Tensor]

      Backbone forward.

      :param data: Atomic systems as input
      :type data: DataBatch

      :returns: **embedding** -- Return backbone embeddings for the given input
      :rtype: dict[str->torch.Tensor]



.. py:class:: eSCNEnergyHead(backbone, reduce='sum')

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


   .. py:attribute:: reduce


   .. py:attribute:: energy_block


   .. py:method:: forward(data: torch_geometric.data.batch.Batch, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]

      Head forward.

      :param data: Atomic systems as input
      :type data: DataBatch
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



.. py:class:: eSCNForceHead(backbone)

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


   .. py:attribute:: force_block


   .. py:method:: forward(data: torch_geometric.data.batch.Batch, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]

      Head forward.

      :param data: Atomic systems as input
      :type data: DataBatch
      :param emb: Embeddings of the input as generated by the backbone
      :type emb: dict[str->torch.Tensor]

      :returns: **outputs** -- Return one or more targets generated by this head
      :rtype: dict[str->torch.Tensor]



.. py:class:: LayerBlock(layer_idx: int, sphere_channels: int, hidden_channels: int, edge_channels: int, lmax_list: list[int], mmax_list: list[int], distance_expansion, max_num_elements: int, SO3_grid: fairchem.core.models.escn.so3.SO3_Grid, act)

   Bases: :py:obj:`torch.nn.Module`


   Layer block: Perform one layer (message passing and aggregation) of the GNN

   :param layer_idx: Layer number
   :type layer_idx: int
   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param hidden_channels: Number of hidden channels used during the SO(2) conv
   :type hidden_channels: int
   :param edge_channels: Size of invariant edge embedding
   :type edge_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution
   :param distance_expansion: Function used to compute distance embedding
   :type distance_expansion: func
   :param max_num_elements: Maximum number of atomic numbers
   :type max_num_elements: int
   :param SO3_grid: Class used to convert from grid the spherical harmonic representations
   :type SO3_grid: SO3_grid
   :param act: Non-linear activation function
   :type act: function


   .. py:attribute:: layer_idx


   .. py:attribute:: act


   .. py:attribute:: lmax_list


   .. py:attribute:: mmax_list


   .. py:attribute:: num_resolutions


   .. py:attribute:: sphere_channels


   .. py:attribute:: sphere_channels_all


   .. py:attribute:: SO3_grid


   .. py:attribute:: message_block


   .. py:attribute:: fc1_sphere


   .. py:attribute:: fc2_sphere


   .. py:attribute:: fc3_sphere


   .. py:method:: forward(x, atomic_numbers, edge_distance, edge_index, SO3_edge_rot, mappingReduced)


.. py:class:: MessageBlock(layer_idx: int, sphere_channels: int, hidden_channels: int, edge_channels: int, lmax_list: list[int], mmax_list: list[int], distance_expansion, max_num_elements: int, SO3_grid: fairchem.core.models.escn.so3.SO3_Grid, act)

   Bases: :py:obj:`torch.nn.Module`


   Message block: Perform message passing

   :param layer_idx: Layer number
   :type layer_idx: int
   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param hidden_channels: Number of hidden channels used during the SO(2) conv
   :type hidden_channels: int
   :param edge_channels: Size of invariant edge embedding
   :type edge_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution
   :param distance_expansion: Function used to compute distance embedding
   :type distance_expansion: func
   :param max_num_elements: Maximum number of atomic numbers
   :type max_num_elements: int
   :param SO3_grid: Class used to convert from grid the spherical harmonic representations
   :type SO3_grid: SO3_grid
   :param act: Non-linear activation function
   :type act: function


   .. py:attribute:: layer_idx


   .. py:attribute:: act


   .. py:attribute:: hidden_channels


   .. py:attribute:: sphere_channels


   .. py:attribute:: SO3_grid


   .. py:attribute:: num_resolutions


   .. py:attribute:: lmax_list


   .. py:attribute:: mmax_list


   .. py:attribute:: edge_channels


   .. py:attribute:: edge_block


   .. py:attribute:: so2_block_source


   .. py:attribute:: so2_block_target


   .. py:method:: forward(x, atomic_numbers, edge_distance, edge_index, SO3_edge_rot, mappingReduced)


.. py:class:: SO2Block(sphere_channels: int, hidden_channels: int, edge_channels: int, lmax_list: list[int], mmax_list: list[int], act)

   Bases: :py:obj:`torch.nn.Module`


   SO(2) Block: Perform SO(2) convolutions for all m (orders)

   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param hidden_channels: Number of hidden channels used during the SO(2) conv
   :type hidden_channels: int
   :param edge_channels: Size of invariant edge embedding
   :type edge_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution
   :param act: Non-linear activation function
   :type act: function


   .. py:attribute:: sphere_channels


   .. py:attribute:: hidden_channels


   .. py:attribute:: lmax_list


   .. py:attribute:: mmax_list


   .. py:attribute:: num_resolutions
      :type:  int


   .. py:attribute:: act


   .. py:attribute:: fc1_dist0


   .. py:attribute:: fc1_m0


   .. py:attribute:: fc2_m0


   .. py:attribute:: so2_conv


   .. py:method:: forward(x, x_edge, mappingReduced)


.. py:class:: SO2Conv(m: int, sphere_channels: int, hidden_channels: int, edge_channels: int, lmax_list: list[int], mmax_list: list[int], act)

   Bases: :py:obj:`torch.nn.Module`


   SO(2) Conv: Perform an SO(2) convolution

   :param m: Order of the spherical harmonic coefficients
   :type m: int
   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param hidden_channels: Number of hidden channels used during the SO(2) conv
   :type hidden_channels: int
   :param edge_channels: Size of invariant edge embedding
   :type edge_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution
   :param act: Non-linear activation function
   :type act: function


   .. py:attribute:: hidden_channels


   .. py:attribute:: lmax_list


   .. py:attribute:: mmax_list


   .. py:attribute:: sphere_channels


   .. py:attribute:: num_resolutions
      :type:  int


   .. py:attribute:: m


   .. py:attribute:: act


   .. py:attribute:: fc1_dist


   .. py:attribute:: fc1_r


   .. py:attribute:: fc2_r


   .. py:attribute:: fc1_i


   .. py:attribute:: fc2_i


   .. py:method:: forward(x_m, x_edge) -> torch.Tensor


.. py:class:: EdgeBlock(edge_channels, distance_expansion, max_num_elements, act)

   Bases: :py:obj:`torch.nn.Module`


   Edge Block: Compute invariant edge representation from edge diatances and atomic numbers

   :param edge_channels: Size of invariant edge embedding
   :type edge_channels: int
   :param distance_expansion: Function used to compute distance embedding
   :type distance_expansion: func
   :param max_num_elements: Maximum number of atomic numbers
   :type max_num_elements: int
   :param act: Non-linear activation function
   :type act: function


   .. py:attribute:: in_channels


   .. py:attribute:: distance_expansion


   .. py:attribute:: act


   .. py:attribute:: edge_channels


   .. py:attribute:: max_num_elements


   .. py:attribute:: fc1_dist


   .. py:attribute:: source_embedding


   .. py:attribute:: target_embedding


   .. py:attribute:: fc1_edge_attr


   .. py:method:: forward(edge_distance, source_element, target_element)


.. py:class:: EnergyBlock(num_channels: int, num_sphere_samples: int, act)

   Bases: :py:obj:`torch.nn.Module`


   Energy Block: Output block computing the energy

   :param num_channels: Number of channels
   :type num_channels: int
   :param num_sphere_samples: Number of samples used to approximate the integral on the sphere
   :type num_sphere_samples: int
   :param act: Non-linear activation function
   :type act: function


   .. py:attribute:: num_channels


   .. py:attribute:: num_sphere_samples


   .. py:attribute:: act


   .. py:attribute:: fc1


   .. py:attribute:: fc2


   .. py:attribute:: fc3


   .. py:method:: forward(x_pt) -> torch.Tensor


.. py:class:: ForceBlock(num_channels: int, num_sphere_samples: int, act)

   Bases: :py:obj:`torch.nn.Module`


   Force Block: Output block computing the per atom forces

   :param num_channels: Number of channels
   :type num_channels: int
   :param num_sphere_samples: Number of samples used to approximate the integral on the sphere
   :type num_sphere_samples: int
   :param act: Non-linear activation function
   :type act: function


   .. py:attribute:: num_channels


   .. py:attribute:: num_sphere_samples


   .. py:attribute:: act


   .. py:attribute:: fc1


   .. py:attribute:: fc2


   .. py:attribute:: fc3


   .. py:method:: forward(x_pt, sphere_points) -> torch.Tensor


