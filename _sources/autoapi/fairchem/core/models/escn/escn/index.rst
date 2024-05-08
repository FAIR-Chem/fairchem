:py:mod:`fairchem.core.models.escn.escn`
========================================

.. py:module:: fairchem.core.models.escn.escn

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.models.escn.escn.eSCN
   fairchem.core.models.escn.escn.LayerBlock
   fairchem.core.models.escn.escn.MessageBlock
   fairchem.core.models.escn.escn.SO2Block
   fairchem.core.models.escn.escn.SO2Conv
   fairchem.core.models.escn.escn.EdgeBlock
   fairchem.core.models.escn.escn.EnergyBlock
   fairchem.core.models.escn.escn.ForceBlock




.. py:class:: eSCN(num_atoms: int, bond_feat_dim: int, num_targets: int, use_pbc: bool = True, regress_forces: bool = True, otf_graph: bool = False, max_neighbors: int = 40, cutoff: float = 8.0, max_num_elements: int = 90, num_layers: int = 8, lmax_list: list[int] | None = None, mmax_list: list[int] | None = None, sphere_channels: int = 128, hidden_channels: int = 256, edge_channels: int = 128, use_grid: bool = True, num_sphere_samples: int = 128, distance_function: str = 'gaussian', basis_width_scalar: float = 1.0, distance_resolution: float = 0.02, show_timing_info: bool = False)


   Bases: :py:obj:`fairchem.core.models.base.BaseModel`

   Equivariant Spherical Channel Network
   Paper: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs


   :param use_pbc: Use periodic boundary conditions
   :type use_pbc: bool
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

   .. py:property:: num_params
      :type: int


   .. py:method:: forward(data)


   .. py:method:: _init_edge_rot_mat(data, edge_index, edge_distance_vec)



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

   .. py:method:: forward(x_pt, sphere_points) -> torch.Tensor



