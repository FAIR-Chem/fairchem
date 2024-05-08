:py:mod:`fairchem.core.models.equiformer_v2.radial_function`
============================================================

.. py:module:: fairchem.core.models.equiformer_v2.radial_function


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.models.equiformer_v2.radial_function.RadialFunction




.. py:class:: RadialFunction(channels_list)


   Bases: :py:obj:`torch.nn.Module`

   Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels

   .. py:method:: forward(inputs)



