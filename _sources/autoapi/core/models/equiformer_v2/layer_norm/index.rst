core.models.equiformer_v2.layer_norm
====================================

.. py:module:: core.models.equiformer_v2.layer_norm

.. autoapi-nested-parse::

   1. Normalize features of shape (N, sphere_basis, C),
   with sphere_basis = (lmax + 1) ** 2.

   2. The difference from `layer_norm.py` is that all type-L vectors have
   the same number of channels and input features are of shape (N, sphere_basis, C).



Classes
-------

.. autoapisummary::

   core.models.equiformer_v2.layer_norm.EquivariantLayerNormArray
   core.models.equiformer_v2.layer_norm.EquivariantLayerNormArraySphericalHarmonics
   core.models.equiformer_v2.layer_norm.EquivariantRMSNormArraySphericalHarmonics
   core.models.equiformer_v2.layer_norm.EquivariantRMSNormArraySphericalHarmonicsV2
   core.models.equiformer_v2.layer_norm.EquivariantDegreeLayerScale


Functions
---------

.. autoapisummary::

   core.models.equiformer_v2.layer_norm.get_normalization_layer
   core.models.equiformer_v2.layer_norm.get_l_to_all_m_expand_index


Module Contents
---------------

.. py:function:: get_normalization_layer(norm_type: str, lmax: int, num_channels: int, eps: float = 1e-05, affine: bool = True, normalization: str = 'component')

.. py:function:: get_l_to_all_m_expand_index(lmax: int)

.. py:class:: EquivariantLayerNormArray(lmax: int, num_channels: int, eps: float = 1e-05, affine: bool = True, normalization: str = 'component')

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


   .. py:attribute:: lmax


   .. py:attribute:: num_channels


   .. py:attribute:: eps


   .. py:attribute:: affine


   .. py:attribute:: normalization


   .. py:method:: __repr__() -> str


   .. py:method:: forward(node_input)

      Assume input is of shape [N, sphere_basis, C]



.. py:class:: EquivariantLayerNormArraySphericalHarmonics(lmax: int, num_channels: int, eps: float = 1e-05, affine: bool = True, normalization: str = 'component', std_balance_degrees: bool = True)

   Bases: :py:obj:`torch.nn.Module`


   1. Normalize over L = 0.
   2. Normalize across all m components from degrees L > 0.
   3. Do not normalize separately for different L (L > 0).


   .. py:attribute:: lmax


   .. py:attribute:: num_channels


   .. py:attribute:: eps


   .. py:attribute:: affine


   .. py:attribute:: std_balance_degrees


   .. py:attribute:: norm_l0


   .. py:attribute:: normalization


   .. py:method:: __repr__() -> str


   .. py:method:: forward(node_input)

      Assume input is of shape [N, sphere_basis, C]



.. py:class:: EquivariantRMSNormArraySphericalHarmonics(lmax: int, num_channels: int, eps: float = 1e-05, affine: bool = True, normalization: str = 'component')

   Bases: :py:obj:`torch.nn.Module`


   1. Normalize across all m components from degrees L >= 0.


   .. py:attribute:: lmax


   .. py:attribute:: num_channels


   .. py:attribute:: eps


   .. py:attribute:: affine


   .. py:attribute:: normalization


   .. py:method:: __repr__() -> str


   .. py:method:: forward(node_input)

      Assume input is of shape [N, sphere_basis, C]



.. py:class:: EquivariantRMSNormArraySphericalHarmonicsV2(lmax: int, num_channels: int, eps: float = 1e-05, affine: bool = True, normalization: str = 'component', centering: bool = True, std_balance_degrees: bool = True)

   Bases: :py:obj:`torch.nn.Module`


   1. Normalize across all m components from degrees L >= 0.
   2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.


   .. py:attribute:: lmax


   .. py:attribute:: num_channels


   .. py:attribute:: eps


   .. py:attribute:: affine


   .. py:attribute:: centering


   .. py:attribute:: std_balance_degrees


   .. py:attribute:: normalization


   .. py:method:: __repr__() -> str


   .. py:method:: forward(node_input)

      Assume input is of shape [N, sphere_basis, C]



.. py:class:: EquivariantDegreeLayerScale(lmax: int, num_channels: int, scale_factor: float = 2.0)

   Bases: :py:obj:`torch.nn.Module`


   1. Similar to Layer Scale used in CaiT (Going Deeper With Image Transformers (ICCV'21)), we scale the output of both attention and FFN.
   2. For degree L > 0, we scale down the square root of 2 * L, which is to emulate halving the number of channels when using higher L.


   .. py:attribute:: lmax


   .. py:attribute:: num_channels


   .. py:attribute:: scale_factor


   .. py:attribute:: affine_weight


   .. py:method:: __repr__() -> str


   .. py:method:: forward(node_input)


