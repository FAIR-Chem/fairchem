:py:mod:`ocpmodels.modules.scaling`
===================================

.. py:module:: ocpmodels.modules.scaling


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   compat/index.rst
   fit/index.rst
   scale_factor/index.rst
   util/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.modules.scaling.ScaleFactor




.. py:class:: ScaleFactor(name: Optional[str] = None, enforce_consistency: bool = True)


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
      :type: Optional[str]

      

   .. py:attribute:: index_fn
      :type: Optional[IndexFn]

      

   .. py:attribute:: stats
      :type: Optional[_Stats]

      

   .. py:method:: _enforce_consistency(state_dict, prefix, _local_metadata, _strict, _missing_keys, _unexpected_keys, _error_msgs) -> None


   .. py:method:: reset_() -> None


   .. py:method:: set_(scale: Union[float, torch.Tensor]) -> None


   .. py:method:: initialize_(*, index_fn: Optional[IndexFn] = None) -> None


   .. py:method:: fit_context_()


   .. py:method:: fit_()


   .. py:method:: _observe(x: torch.Tensor, ref: Optional[torch.Tensor] = None) -> None


   .. py:method:: forward(x: torch.Tensor, *, ref: Optional[torch.Tensor] = None) -> torch.Tensor



