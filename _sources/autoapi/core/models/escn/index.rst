core.models.escn
================

.. py:module:: core.models.escn


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/models/escn/escn/index
   /autoapi/core/models/escn/escn_exportable/index
   /autoapi/core/models/escn/so3/index
   /autoapi/core/models/escn/so3_exportable/index


Classes
-------

.. autoapisummary::

   core.models.escn.eSCN


Package Contents
----------------

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



