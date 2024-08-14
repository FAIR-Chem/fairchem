core.models.finetune_hydra
==========================

.. py:module:: core.models.finetune_hydra


Attributes
----------

.. autoapisummary::

   core.models.finetune_hydra.FTHYDRA_NAME


Classes
-------

.. autoapisummary::

   core.models.finetune_hydra.FineTuneMode
   core.models.finetune_hydra.FTConfig
   core.models.finetune_hydra.FineTuneHydra


Functions
---------

.. autoapisummary::

   core.models.finetune_hydra.get_model_config_from_checkpoint
   core.models.finetune_hydra.load_hydra_model


Module Contents
---------------

.. py:data:: FTHYDRA_NAME
   :value: 'finetune_hydra'


.. py:class:: FineTuneMode(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access::

   >>> Color.RED
   <Color.RED: 1>

   - value lookup:

   >>> Color(1)
   <Color.RED: 1>

   - name lookup:

   >>> Color['RED']
   <Color.RED: 1>

   Enumerations can be iterated over, and know how many members they have:

   >>> len(Color)
   3

   >>> list(Color)
   [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

   Methods can be added to enumerations, and members can have their own
   attributes -- see the documentation for details.


   .. py:attribute:: DATA_ONLY
      :value: 1



   .. py:attribute:: RETAIN_BACKBONE_ONLY
      :value: 2



.. py:function:: get_model_config_from_checkpoint(checkpoint_path: str) -> dict

.. py:function:: load_hydra_model(checkpoint_path: str) -> fairchem.core.models.base.HydraInterface

.. py:class:: FTConfig(config: dict)

   .. py:attribute:: FT_CONFIG_NAME
      :value: 'finetune_config'



   .. py:attribute:: STARTING_CHECKPOINT
      :value: 'starting_checkpoint'



   .. py:attribute:: STARTING_MODEL
      :value: 'starting_model'



   .. py:attribute:: MODE
      :value: 'mode'



   .. py:attribute:: HEADS
      :value: 'heads'



   .. py:attribute:: config


   .. py:attribute:: _mode


   .. py:method:: load_model() -> torch.nn.Module


   .. py:method:: get_standalone_config() -> dict


   .. py:property:: mode
      :type: FineTuneMode



   .. py:property:: head_config
      :type: dict



.. py:class:: FineTuneHydra(finetune_config: dict)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.HydraInterface`


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


   .. py:attribute:: ft_config


   .. py:attribute:: hydra_model
      :type:  fairchem.core.models.base.HydraInterface


   .. py:attribute:: backbone
      :type:  fairchem.core.models.base.BackboneInterface


   .. py:method:: forward(data: torch_geometric.data.Batch)


   .. py:method:: get_backbone() -> fairchem.core.models.base.BackboneInterface


   .. py:method:: get_heads() -> dict[str, fairchem.core.models.base.HeadInterface]


