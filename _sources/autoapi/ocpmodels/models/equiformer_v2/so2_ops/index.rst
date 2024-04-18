:py:mod:`ocpmodels.models.equiformer_v2.so2_ops`
================================================

.. py:module:: ocpmodels.models.equiformer_v2.so2_ops


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.equiformer_v2.so2_ops.SO2_m_Convolution
   ocpmodels.models.equiformer_v2.so2_ops.SO2_Convolution
   ocpmodels.models.equiformer_v2.so2_ops.SO2_Linear




.. py:class:: SO2_m_Convolution(m: int, sphere_channels: int, m_output_channels: int, lmax_list: List[int], mmax_list: List[int])


   Bases: :py:obj:`torch.nn.Module`

   SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

   :param m: Order of the spherical harmonic coefficients
   :type m: int
   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param m_output_channels: Number of output channels used during the SO(2) conv
   :type m_output_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution

   .. py:method:: forward(x_m)



.. py:class:: SO2_Convolution(sphere_channels: int, m_output_channels: int, lmax_list: List[int], mmax_list: List[int], mappingReduced, internal_weights: bool = True, edge_channels_list: Optional[List[int]] = None, extra_m0_output_channels: Optional[int] = None)


   Bases: :py:obj:`torch.nn.Module`

   SO(2) Block: Perform SO(2) convolutions for all m (orders)

   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param m_output_channels: Number of output channels used during the SO(2) conv
   :type m_output_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution
   :param mappingReduced: Used to extract a subset of m components
   :type mappingReduced: CoefficientMappingModule
   :param internal_weights: If True, not using radial function to multiply inputs features
   :type internal_weights: bool
   :param edge_channels_list (list: int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
   :param extra_m0_output_channels: If not None, return `out_embedding` (SO3_Embedding) and `extra_m0_features` (Tensor).
   :type extra_m0_output_channels: int

   .. py:method:: forward(x, x_edge)



.. py:class:: SO2_Linear(sphere_channels: int, m_output_channels: int, lmax_list: List[int], mmax_list: List[int], mappingReduced, internal_weights: bool = False, edge_channels_list: Optional[List[int]] = None)


   Bases: :py:obj:`torch.nn.Module`

   SO(2) Linear: Perform SO(2) linear for all m (orders).

   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param m_output_channels: Number of output channels used during the SO(2) conv
   :type m_output_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution
   :param mappingReduced: Used to extract a subset of m components
   :type mappingReduced: CoefficientMappingModule
   :param internal_weights: If True, not using radial function to multiply inputs features
   :type internal_weights: bool
   :param edge_channels_list (list: int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].

   .. py:method:: forward(x, x_edge)



