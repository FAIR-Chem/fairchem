core.models
===========

.. py:module:: core.models


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/models/base/index
   /autoapi/core/models/dimenet_plus_plus/index
   /autoapi/core/models/equiformer_v2/index
   /autoapi/core/models/escn/index
   /autoapi/core/models/gemnet/index
   /autoapi/core/models/gemnet_gp/index
   /autoapi/core/models/gemnet_oc/index
   /autoapi/core/models/model_registry/index
   /autoapi/core/models/painn/index
   /autoapi/core/models/schnet/index
   /autoapi/core/models/scn/index
   /autoapi/core/models/utils/index


Attributes
----------

.. autoapisummary::

   core.models.available_pretrained_models


Functions
---------

.. autoapisummary::

   core.models.model_name_to_local_file


Package Contents
----------------

.. py:data:: available_pretrained_models

.. py:function:: model_name_to_local_file(model_name: str, local_cache: str | pathlib.Path) -> str

   Download a pretrained checkpoint if it does not exist already

   :param model_name: the model name. See available_pretrained_checkpoints.
   :type model_name: str
   :param local_cache: path to local cache directory
   :type local_cache: str or Path

   :returns: local path to checkpoint file
   :rtype: str


