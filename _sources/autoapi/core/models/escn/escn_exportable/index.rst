core.models.escn.escn_exportable
================================

.. py:module:: core.models.escn.escn_exportable

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.escn.escn_exportable.eSCN
   core.models.escn.escn_exportable.LayerBlock
   core.models.escn.escn_exportable.MessageBlock
   core.models.escn.escn_exportable.SO2Block
   core.models.escn.escn_exportable.SO2Conv
   core.models.escn.escn_exportable.EdgeBlock
   core.models.escn.escn_exportable.EnergyBlock
   core.models.escn.escn_exportable.ForceBlock


Module Contents
---------------

.. py:class:: eSCN(max_neighbors: int = 300, cutoff: float = 8.0, max_num_elements: int = 100, num_layers: int = 8, lmax: int = 4, mmax: int = 2, sphere_channels: int = 128, hidden_channels: int = 256, edge_channels: int = 128, num_sphere_samples: int = 128, distance_function: str = 'gaussian', basis_width_scalar: float = 1.0, distance_resolution: float = 0.02, resolution: int | None = None, compile: bool = False, export: bool = False, rescale_grid: bool = False)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.GraphModelMixin`


   Equivariant Spherical Channel Network
   Paper: Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs


   :param max_neighbors: Max neighbors to take per node, when using the graph generation
   :type max_neighbors: int
   :param cutoff: Maximum distance between nieghboring atoms in Angstroms
   :type cutoff: float
   :param max_num_elements: Maximum atomic number
   :type max_num_elements: int
   :param num_layers: Number of layers in the GNN
   :type num_layers: int
   :param lmax: maximum degree of the spherical harmonics (1 to 10)
   :type lmax: int
   :param mmax: maximum order of the spherical harmonics (0 to lmax)
   :type mmax: int
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
   :param compile: use torch.compile on the forward
   :type compile: bool
   :param export: use the exportable version of the module
   :type export: bool


   .. py:attribute:: max_neighbors


   .. py:attribute:: cutoff


   .. py:attribute:: max_num_elements


   .. py:attribute:: hidden_channels


   .. py:attribute:: num_layers


   .. py:attribute:: num_sphere_samples


   .. py:attribute:: sphere_channels


   .. py:attribute:: edge_channels


   .. py:attribute:: distance_resolution


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: basis_width_scalar


   .. py:attribute:: distance_function


   .. py:attribute:: compile

      Compile this Module's forward using :func:`torch.compile`.

      This Module's `__call__` method is compiled and all arguments are passed as-is
      to :func:`torch.compile`.

      See :func:`torch.compile` for details on the arguments for this function.


   .. py:attribute:: export


   .. py:attribute:: rescale_grid


   .. py:attribute:: act


   .. py:attribute:: sphere_embedding


   .. py:attribute:: num_gaussians


   .. py:attribute:: SO3_grid


   .. py:attribute:: layer_blocks


   .. py:attribute:: energy_block


   .. py:attribute:: force_block


   .. py:attribute:: sphere_points


   .. py:attribute:: sphharm_weights
      :type:  torch.nn.Parameter


   .. py:attribute:: sph_feature_size


   .. py:method:: forward_trainable(data: torch_geometric.data.batch.Batch) -> dict[str, torch.Tensor]


   .. py:method:: forward(pos: torch.Tensor, batch_idx: torch.Tensor, natoms: torch.Tensor, atomic_numbers: torch.Tensor, edge_index: torch.Tensor, edge_distance: torch.Tensor, edge_distance_vec: torch.Tensor) -> list[torch.Tensor]

      N: num atoms
      N: batch size
      E: num edges

      pos: [N, 3] atom positions
      batch_idx: [N] batch index of each atom
      natoms: [B] number of atoms in each batch
      atomic_numbers: [N] atomic number per atom
      edge_index: [2, E] edges between source and target atoms
      edge_distance: [E] cartesian distance for each edge
      edge_distance_vec: [E, 3] direction vector of edges (includes pbc)



   .. py:method:: _init_edge_rot_mat(edge_distance_vec)


   .. py:property:: num_params
      :type: int



.. py:class:: LayerBlock(layer_idx: int, sphere_channels: int, hidden_channels: int, edge_channels: int, lmax: int, mmax: int, distance_expansion, max_num_elements: int, SO3_grid: fairchem.core.models.escn.so3_exportable.SO3_Grid, act)

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
   :param lmax:
   :type lmax: int)                  degrees (l
   :param mmax: orders (m) for each resolution
   :type mmax: int
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


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: sphere_channels


   .. py:attribute:: SO3_grid


   .. py:attribute:: message_block


   .. py:attribute:: fc1_sphere


   .. py:attribute:: fc2_sphere


   .. py:attribute:: fc3_sphere


   .. py:method:: forward(x: torch.Tensor, atomic_numbers: torch.Tensor, edge_distance: torch.Tensor, edge_index: torch.Tensor, wigner: torch.Tensor) -> torch.Tensor


.. py:class:: MessageBlock(layer_idx: int, sphere_channels: int, hidden_channels: int, edge_channels: int, lmax: int, mmax: int, distance_expansion, max_num_elements: int, SO3_grid: fairchem.core.models.escn.so3_exportable.SO3_Grid, act)

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
   :param lmax: degrees (l) for each resolution
   :type lmax: int
   :param mmax: orders (m) for each resolution
   :type mmax: int
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


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: edge_channels


   .. py:attribute:: out_mask


   .. py:attribute:: edge_block


   .. py:attribute:: so2_block_source


   .. py:attribute:: so2_block_target


   .. py:method:: forward(x: torch.Tensor, atomic_numbers: torch.Tensor, edge_distance: torch.Tensor, edge_index: torch.Tensor, wigner: torch.Tensor) -> torch.Tensor


.. py:class:: SO2Block(sphere_channels: int, hidden_channels: int, edge_channels: int, lmax: int, mmax: int, act)

   Bases: :py:obj:`torch.nn.Module`


   SO(2) Block: Perform SO(2) convolutions for all m (orders)

   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param hidden_channels: Number of hidden channels used during the SO(2) conv
   :type hidden_channels: int
   :param edge_channels: Size of invariant edge embedding
   :type edge_channels: int
   :param lmax: degrees (l) for each resolution
   :type lmax: int
   :param mmax: orders (m) for each resolution
   :type mmax: int
   :param act: Non-linear activation function
   :type act: function


   .. py:attribute:: sphere_channels


   .. py:attribute:: hidden_channels


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: act


   .. py:attribute:: mappingReduced


   .. py:attribute:: fc1_dist0


   .. py:attribute:: fc1_m0


   .. py:attribute:: fc2_m0


   .. py:attribute:: so2_conv


   .. py:method:: forward(x: torch.Tensor, x_edge: torch.Tensor)


.. py:class:: SO2Conv(m: int, sphere_channels: int, hidden_channels: int, edge_channels: int, lmax: int, mmax: int, act)

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
   :param lmax: degrees (l) for each resolution
   :type lmax: int
   :param mmax: orders (m) for each resolution
   :type mmax: int
   :param act: Non-linear activation function
   :type act: function


   .. py:attribute:: hidden_channels


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: sphere_channels


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


