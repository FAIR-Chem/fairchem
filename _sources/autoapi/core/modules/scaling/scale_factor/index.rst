:py:mod:`core.modules.scaling.scale_factor`
===========================================

.. py:module:: core.modules.scaling.scale_factor


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   core.modules.scaling.scale_factor._Stats
   core.modules.scaling.scale_factor.ScaleFactor



Functions
~~~~~~~~~

.. autoapisummary::

   core.modules.scaling.scale_factor._check_consistency



Attributes
~~~~~~~~~~

.. autoapisummary::

   core.modules.scaling.scale_factor.IndexFn


.. py:class:: _Stats


   Bases: :py:obj:`TypedDict`

   dict() -> new empty dictionary
   dict(mapping) -> new dictionary initialized from a mapping object's
       (key, value) pairs
   dict(iterable) -> new dictionary initialized as if via:
       d = {}
       for k, v in iterable:
           d[k] = v
   dict(**kwargs) -> new dictionary initialized with the name=value pairs
       in the keyword argument list.  For example:  dict(one=1, two=2)

   .. py:attribute:: variance_in
      :type: float

      

   .. py:attribute:: variance_out
      :type: float

      

   .. py:attribute:: n_samples
      :type: int

      


.. py:data:: IndexFn

   

.. py:function:: _check_consistency(old: torch.Tensor, new: torch.Tensor, key: str) -> None


.. py:class:: ScaleFactor(name: str | None = None, enforce_consistency: bool = True)


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

   .. py:property:: fitted
      :type: bool


   .. py:attribute:: scale_factor
      :type: torch.Tensor

      

   .. py:attribute:: name
      :type: str | None

      

   .. py:attribute:: index_fn
      :type: IndexFn | None

      

   .. py:attribute:: stats
      :type: _Stats | None

      

   .. py:method:: _enforce_consistency(state_dict, prefix, _local_metadata, _strict, _missing_keys, _unexpected_keys, _error_msgs) -> None


   .. py:method:: reset_() -> None


   .. py:method:: set_(scale: float | torch.Tensor) -> None


   .. py:method:: initialize_(*, index_fn: IndexFn | None = None) -> None


   .. py:method:: fit_context_()


   .. py:method:: fit_()


   .. py:method:: _observe(x: torch.Tensor, ref: torch.Tensor | None = None) -> None


   .. py:method:: forward(x: torch.Tensor, *, ref: torch.Tensor | None = None) -> torch.Tensor



