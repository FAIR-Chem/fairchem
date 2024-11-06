core.models.equiformer_v2.so2_ops
=================================

.. py:module:: core.models.equiformer_v2.so2_ops


Classes
-------

.. autoapisummary::

   core.models.equiformer_v2.so2_ops.SO2_m_Convolution
   core.models.equiformer_v2.so2_ops.SO2_Convolution
   core.models.equiformer_v2.so2_ops.SO2_Linear


Module Contents
---------------

.. py:class:: SO2_m_Convolution(m: int, sphere_channels: int, m_output_channels: int, lmax_list: list[int], mmax_list: list[int])

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


   .. py:attribute:: m


   .. py:attribute:: sphere_channels


   .. py:attribute:: m_output_channels


   .. py:attribute:: lmax_list


   .. py:attribute:: mmax_list


   .. py:attribute:: num_resolutions
      :type:  int


   .. py:attribute:: fc


   .. py:method:: forward(x_m)


.. py:class:: SO2_Convolution(sphere_channels: int, m_output_channels: int, lmax_list: list[int], mmax_list: list[int], mappingReduced, internal_weights: bool = True, edge_channels_list: list[int] | None = None, extra_m0_output_channels: int | None = None)

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


   .. py:attribute:: sphere_channels


   .. py:attribute:: m_output_channels


   .. py:attribute:: lmax_list


   .. py:attribute:: mmax_list


   .. py:attribute:: mappingReduced


   .. py:attribute:: num_resolutions


   .. py:attribute:: internal_weights


   .. py:attribute:: edge_channels_list


   .. py:attribute:: extra_m0_output_channels


   .. py:attribute:: fc_m0


   .. py:attribute:: so2_m_conv


   .. py:attribute:: rad_func
      :value: None



   .. py:method:: forward(x, x_edge)


.. py:class:: SO2_Linear(sphere_channels: int, m_output_channels: int, lmax_list: list[int], mmax_list: list[int], mappingReduced, internal_weights: bool = False, edge_channels_list: list[int] | None = None)

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


   .. py:attribute:: sphere_channels


   .. py:attribute:: m_output_channels


   .. py:attribute:: lmax_list


   .. py:attribute:: mmax_list


   .. py:attribute:: mappingReduced


   .. py:attribute:: internal_weights


   .. py:attribute:: edge_channels_list


   .. py:attribute:: num_resolutions


   .. py:attribute:: fc_m0


   .. py:attribute:: so2_m_fc


   .. py:attribute:: rad_func
      :value: None



   .. py:method:: forward(x, x_edge)


