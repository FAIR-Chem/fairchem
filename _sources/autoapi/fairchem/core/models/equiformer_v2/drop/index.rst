:py:mod:`fairchem.core.models.equiformer_v2.drop`
=================================================

.. py:module:: fairchem.core.models.equiformer_v2.drop

.. autoapi-nested-parse::

   Add `extra_repr` into DropPath implemented by timm
   for displaying more info.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.models.equiformer_v2.drop.DropPath
   fairchem.core.models.equiformer_v2.drop.GraphDropPath
   fairchem.core.models.equiformer_v2.drop.EquivariantDropout
   fairchem.core.models.equiformer_v2.drop.EquivariantScalarsDropout
   fairchem.core.models.equiformer_v2.drop.EquivariantDropoutArraySphericalHarmonics



Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.models.equiformer_v2.drop.drop_path



.. py:function:: drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor

   Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
   This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
   the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
   See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
   changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
   'survival rate' as the argument.


.. py:class:: DropPath(drop_prob: float)


   Bases: :py:obj:`torch.nn.Module`

   Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


   .. py:method:: extra_repr() -> str

      Set the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: GraphDropPath(drop_prob: float)


   Bases: :py:obj:`torch.nn.Module`

   Consider batch for graph data when dropping paths.

   .. py:method:: forward(x: torch.Tensor, batch) -> torch.Tensor


   .. py:method:: extra_repr() -> str

      Set the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: EquivariantDropout(irreps, drop_prob: float)


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

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor



.. py:class:: EquivariantScalarsDropout(irreps, drop_prob: float)


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

   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


   .. py:method:: extra_repr() -> str

      Set the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: EquivariantDropoutArraySphericalHarmonics(drop_prob: float, drop_graph: bool = False)


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

   .. py:method:: forward(x: torch.Tensor, batch=None) -> torch.Tensor


   .. py:method:: extra_repr() -> str

      Set the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



