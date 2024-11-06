core.models.escn.so3_exportable
===============================

.. py:module:: core.models.escn.so3_exportable


Classes
-------

.. autoapisummary::

   core.models.escn.so3_exportable.CoefficientMapping
   core.models.escn.so3_exportable.SO3_Grid


Functions
---------

.. autoapisummary::

   core.models.escn.so3_exportable.wigner_D
   core.models.escn.so3_exportable._z_rot_mat
   core.models.escn.so3_exportable.rotation_to_wigner


Module Contents
---------------

.. py:function:: wigner_D(lv: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, _Jd: list[torch.Tensor]) -> torch.Tensor

.. py:function:: _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor

.. py:function:: rotation_to_wigner(edge_rot_mat: torch.Tensor, start_lmax: int, end_lmax: int, Jd: list[torch.Tensor]) -> torch.Tensor

.. py:class:: CoefficientMapping(lmax_list, mmax_list)

   Bases: :py:obj:`torch.nn.Module`


   Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

   :param lmax_list (list: int):   List of maximum degree of the spherical harmonics
   :param mmax_list (list: int):   List of maximum order of the spherical harmonics
   :param use_rotate_inv_rescale: Whether to pre-compute inverse rotation rescale matrices
   :type use_rotate_inv_rescale: bool


   .. py:attribute:: lmax_list


   .. py:attribute:: mmax_list


   .. py:attribute:: num_resolutions


   .. py:attribute:: res_size


   .. py:attribute:: m_size


   .. py:method:: complex_idx(m, lmax, m_complex, l_harmonic)

      Add `m_complex` and `l_harmonic` to the input arguments
      since we cannot use `self.m_complex`.



   .. py:method:: pre_compute_coefficient_idx()

      Pre-compute the results of `coefficient_idx()` and access them with `prepare_coefficient_idx()`



   .. py:method:: prepare_coefficient_idx()

      Construct a list of buffers



   .. py:method:: coefficient_idx(lmax: int, mmax: int)


   .. py:method:: pre_compute_rotate_inv_rescale()


   .. py:method:: __repr__()


.. py:class:: SO3_Grid(lmax: int, mmax: int, normalization: str = 'integral', resolution: int | None = None, rescale: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   Helper functions for grid representation of the irreps

   :param lmax: Maximum degree of the spherical harmonics
   :type lmax: int
   :param mmax: Maximum order of the spherical harmonics
   :type mmax: int


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: lat_resolution


   .. py:attribute:: mapping


   .. py:method:: get_to_grid_mat(device=None)


   .. py:method:: get_from_grid_mat(device=None)


   .. py:method:: to_grid(embedding, lmax: int, mmax: int)


   .. py:method:: from_grid(grid, lmax: int, mmax: int)


