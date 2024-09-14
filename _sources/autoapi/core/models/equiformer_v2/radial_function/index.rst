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


   .. py:attribute:: modules
      :value: []


      Return an iterator over all modules in the network.

      :Yields: *Module* -- a module in the network

      .. note::

         Duplicate modules are returned only once. In the following
         example, ``l`` will be returned only once.

      Example::

          >>> l = nn.Linear(2, 2)
          >>> net = nn.Sequential(l, l)
          >>> for idx, m in enumerate(net.modules()):
          ...     print(idx, '->', m)

          0 -> Sequential(
            (0): Linear(in_features=2, out_features=2, bias=True)
            (1): Linear(in_features=2, out_features=2, bias=True)
          )
          1 -> Linear(in_features=2, out_features=2, bias=True)


   .. py:attribute:: input_channels


   .. py:attribute:: net


   .. py:method:: forward(inputs)


