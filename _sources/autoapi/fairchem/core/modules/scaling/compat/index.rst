:py:mod:`fairchem.core.modules.scaling.compat`
==============================================

.. py:module:: fairchem.core.modules.scaling.compat


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.modules.scaling.compat._load_scale_dict
   fairchem.core.modules.scaling.compat.load_scales_compat



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.core.modules.scaling.compat.ScaleDict


.. py:data:: ScaleDict

   

.. py:function:: _load_scale_dict(scale_file: str | ScaleDict | None)

   Loads scale factors from either:
   - a JSON file mapping scale factor names to scale values
   - a python dictionary pickled object (loaded using `torch.load`) mapping scale factor names to scale values
   - a dictionary mapping scale factor names to scale values


.. py:function:: load_scales_compat(module: torch.nn.Module, scale_file: str | ScaleDict | None) -> None


