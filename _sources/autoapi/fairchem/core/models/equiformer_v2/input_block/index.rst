:py:mod:`fairchem.core.models.equiformer_v2.input_block`
========================================================

.. py:module:: fairchem.core.models.equiformer_v2.input_block


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.models.equiformer_v2.input_block.EdgeDegreeEmbedding




.. py:class:: EdgeDegreeEmbedding(sphere_channels: int, lmax_list: list[int], mmax_list: list[int], SO3_rotation, mappingReduced, max_num_elements: int, edge_channels_list, use_atom_edge_embedding: bool, rescale_factor)


   Bases: :py:obj:`torch.nn.Module`

   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution
   :param SO3_rotation (list: SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
   :param mappingReduced: Class to convert l and m indices once node embedding is rotated
   :type mappingReduced: CoefficientMappingModule
   :param max_num_elements: Maximum number of atomic numbers
   :type max_num_elements: int
   :param edge_channels_list (list: int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                    The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
   :param use_atom_edge_embedding: Whether to use atomic embedding along with relative distance for edge scalar features
   :type use_atom_edge_embedding: bool
   :param rescale_factor: Rescale the sum aggregation
   :type rescale_factor: float

   .. py:method:: forward(atomic_numbers, edge_distance, edge_index)



