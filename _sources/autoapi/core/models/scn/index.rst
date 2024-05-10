:py:mod:`core.models.scn`
=========================

.. py:module:: core.models.scn


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   sampling/index.rst
   scn/index.rst
   smearing/index.rst
   spherical_harmonics/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   core.models.scn.SphericalChannelNetwork




.. py:class:: SphericalChannelNetwork(num_atoms: int, bond_feat_dim: int, num_targets: int, use_pbc: bool = True, regress_forces: bool = True, otf_graph: bool = False, max_num_neighbors: int = 20, cutoff: float = 8.0, max_num_elements: int = 90, num_interactions: int = 8, lmax: int = 6, mmax: int = 1, num_resolutions: int = 2, sphere_channels: int = 128, sphere_channels_reduce: int = 128, hidden_channels: int = 256, num_taps: int = -1, use_grid: bool = True, num_bands: int = 1, num_sphere_samples: int = 128, num_basis_functions: int = 128, distance_function: str = 'gaussian', basis_width_scalar: float = 1.0, distance_resolution: float = 0.02, show_timing_info: bool = False, direct_forces: bool = True)


   Bases: :py:obj:`fairchem.core.models.base.BaseModel`

   Spherical Channel Network
   Paper: Spherical Channels for Modeling Atomic Interactions

   :param use_pbc: Use periodic boundary conditions
   :type use_pbc: bool
   :param regress_forces: Compute forces
   :type regress_forces: bool
   :param otf_graph: Compute graph On The Fly (OTF)
   :type otf_graph: bool
   :param max_num_neighbors: Maximum number of neighbors per atom
   :type max_num_neighbors: int
   :param cutoff: Maximum distance between nieghboring atoms in Angstroms
   :type cutoff: float
   :param max_num_elements: Maximum atomic number
   :type max_num_elements: int
   :param num_interactions: Number of layers in the GNN
   :type num_interactions: int
   :param lmax: Maximum degree of the spherical harmonics (1 to 10)
   :type lmax: int
   :param mmax: Maximum order of the spherical harmonics (0 or 1)
   :type mmax: int
   :param num_resolutions: Number of resolutions used to compute messages, further away atoms has lower resolution (1 or 2)
   :type num_resolutions: int
   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param sphere_channels_reduce: Number of spherical channels used during message passing (downsample or upsample)
   :type sphere_channels_reduce: int
   :param hidden_channels: Number of hidden units in message passing
   :type hidden_channels: int
   :param num_taps: Number of taps or rotations used during message passing (1 or otherwise set automatically based on mmax)
   :type num_taps: int
   :param use_grid: Use non-linear pointwise convolution during aggregation
   :type use_grid: bool
   :param num_bands: Number of bands used during message aggregation for the 1x1 pointwise convolution (1 or 2)
   :type num_bands: int
   :param num_sphere_samples: Number of samples used to approximate the integration of the sphere in the output blocks
   :type num_sphere_samples: int
   :param num_basis_functions: Number of basis functions used for distance and atomic number blocks
   :type num_basis_functions: int
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


   .. py:attribute:: energy_fc1
      :type: torch.nn.Linear

      

   .. py:attribute:: energy_fc2
      :type: torch.nn.Linear

      

   .. py:attribute:: energy_fc3
      :type: torch.nn.Linear

      

   .. py:attribute:: force_fc1
      :type: torch.nn.Linear

      

   .. py:attribute:: force_fc2
      :type: torch.nn.Linear

      

   .. py:attribute:: force_fc3
      :type: torch.nn.Linear

      

   .. py:method:: forward(data)


   .. py:method:: _forward_helper(data)


   .. py:method:: _init_edge_rot_mat(data, edge_index, edge_distance_vec)


   .. py:method:: _rank_edge_distances(edge_distance, edge_index, max_num_neighbors: int) -> torch.Tensor



