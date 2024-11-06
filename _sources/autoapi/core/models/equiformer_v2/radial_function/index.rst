core.models.equiformer_v2.radial_function
=========================================

.. py:module:: core.models.equiformer_v2.radial_function


Classes
-------

.. autoapisummary::

   core.models.equiformer_v2.radial_function.RadialFunction


Module Contents
---------------

.. py:class:: RadialFunction(channels_list)

   Bases: :py:obj:`torch.nn.Module`


   Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels


   .. py:attribute:: net


   .. py:method:: forward(inputs)


