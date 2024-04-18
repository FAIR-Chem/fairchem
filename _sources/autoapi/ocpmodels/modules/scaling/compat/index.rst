:py:mod:`ocpmodels.modules.scaling.compat`
==========================================

.. py:module:: ocpmodels.modules.scaling.compat


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.modules.scaling.compat._load_scale_dict
   ocpmodels.modules.scaling.compat.load_scales_compat



Attributes
~~~~~~~~~~

.. autoapisummary::

   ocpmodels.modules.scaling.compat.ScaleDict


.. py:data:: ScaleDict

   

.. py:function:: _load_scale_dict(scale_file: Optional[Union[str, ScaleDict]])

   Loads scale factors from either:
   - a JSON file mapping scale factor names to scale values
   - a python dictionary pickled object (loaded using `torch.load`) mapping scale factor names to scale values
   - a dictionary mapping scale factor names to scale values


.. py:function:: load_scales_compat(module: torch.nn.Module, scale_file: Optional[Union[str, ScaleDict]]) -> None


