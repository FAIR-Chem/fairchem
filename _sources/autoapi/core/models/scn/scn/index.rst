core.models.scn.scn
===================

.. py:module:: core.models.scn.scn

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.scn.scn.SphericalChannelNetwork
   core.models.scn.scn.EdgeBlock
   core.models.scn.scn.MessageBlock
   core.models.scn.scn.DistanceBlock


Module Contents
---------------

.. py:class:: SphericalChannelNetwork(use_pbc: bool = True, regress_forces: bool = True, otf_graph: bool = False, max_num_neighbors: int = 20, cutoff: float = 8.0, max_num_elements: int = 90, num_interactions: int = 8, lmax: int = 6, mmax: int = 1, num_resolutions: int = 2, sphere_channels: int = 128, sphere_channels_reduce: int = 128, hidden_channels: int = 256, num_taps: int = -1, use_grid: bool = True, num_bands: int = 1, num_sphere_samples: int = 128, num_basis_functions: int = 128, distance_function: str = 'gaussian', basis_width_scalar: float = 1.0, distance_resolution: float = 0.02, show_timing_info: bool = False, direct_forces: bool = True)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.GraphModelMixin`


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


   .. py:attribute:: energy_fc1
      :type:  torch.nn.Linear


   .. py:attribute:: energy_fc2
      :type:  torch.nn.Linear


   .. py:attribute:: energy_fc3
      :type:  torch.nn.Linear


   .. py:attribute:: force_fc1
      :type:  torch.nn.Linear


   .. py:attribute:: force_fc2
      :type:  torch.nn.Linear


   .. py:attribute:: force_fc3
      :type:  torch.nn.Linear


   .. py:attribute:: regress_forces


   .. py:attribute:: use_pbc


   .. py:attribute:: cutoff


   .. py:attribute:: otf_graph


   .. py:attribute:: show_timing_info


   .. py:attribute:: max_num_elements


   .. py:attribute:: hidden_channels


   .. py:attribute:: num_interactions


   .. py:attribute:: num_atoms
      :value: 0



   .. py:attribute:: num_sphere_samples


   .. py:attribute:: sphere_channels


   .. py:attribute:: sphere_channels_reduce


   .. py:attribute:: num_basis_functions


   .. py:attribute:: distance_resolution


   .. py:attribute:: grad_forces
      :value: False



   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: basis_width_scalar


   .. py:attribute:: sphere_basis


   .. py:attribute:: use_grid


   .. py:attribute:: distance_function


   .. py:attribute:: counter
      :value: 0



   .. py:attribute:: act


   .. py:attribute:: sphere_embedding


   .. py:attribute:: num_gaussians


   .. py:attribute:: sphharm_list
      :value: []



   .. py:attribute:: edge_blocks


   .. py:method:: forward(data)


   .. py:method:: _forward_helper(data)


   .. py:method:: _init_edge_rot_mat(data, edge_index, edge_distance_vec)


   .. py:method:: _rank_edge_distances(edge_distance, edge_index, max_num_neighbors: int) -> torch.Tensor


   .. py:property:: num_params
      :type: int



.. py:class:: EdgeBlock(num_resolutions: int, sphere_channels_reduce, hidden_channels_list, cutoff_list, sphharm_list, sphere_channels, distance_expansion, max_num_elements: int, num_basis_functions: int, num_gaussians: int, use_grid: bool, act)

   Bases: :py:obj:`torch.nn.Module`


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


   .. py:attribute:: num_resolutions


   .. py:attribute:: act


   .. py:attribute:: hidden_channels_list


   .. py:attribute:: sphere_channels


   .. py:attribute:: sphere_channels_reduce


   .. py:attribute:: distance_expansion


   .. py:attribute:: cutoff_list


   .. py:attribute:: sphharm_list


   .. py:attribute:: max_num_elements


   .. py:attribute:: num_basis_functions


   .. py:attribute:: use_grid


   .. py:attribute:: num_gaussians


   .. py:attribute:: dist_block


   .. py:attribute:: message_blocks


   .. py:method:: forward(x, atomic_numbers, edge_distance, edge_index, cutoff_index)


.. py:class:: MessageBlock(sphere_channels_reduce, hidden_channels, num_basis_functions, sphharm, act)

   Bases: :py:obj:`torch.nn.Module`


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


   .. py:attribute:: act


   .. py:attribute:: hidden_channels


   .. py:attribute:: sphere_channels_reduce


   .. py:attribute:: sphharm


   .. py:attribute:: fc1_dist


   .. py:attribute:: fc1_edge_proj


   .. py:attribute:: fc1_edge


   .. py:attribute:: fc2_edge


   .. py:method:: forward(x, x_edge, edge_index)


.. py:class:: DistanceBlock(in_channels, num_basis_functions: int, distance_expansion, max_num_elements: int, act)

   Bases: :py:obj:`torch.nn.Module`


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


   .. py:attribute:: in_channels


   .. py:attribute:: distance_expansion


   .. py:attribute:: act


   .. py:attribute:: num_basis_functions


   .. py:attribute:: max_num_elements


   .. py:attribute:: num_edge_channels


   .. py:attribute:: fc1_dist


   .. py:attribute:: source_embedding


   .. py:attribute:: target_embedding


   .. py:attribute:: fc1_edge_attr


   .. py:method:: forward(edge_distance, source_element, target_element)


