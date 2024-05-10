:py:mod:`core.models`
=====================

.. py:module:: core.models


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   equiformer_v2/index.rst
   escn/index.rst
   gemnet/index.rst
   gemnet_gp/index.rst
   gemnet_oc/index.rst
   painn/index.rst
   scn/index.rst
   utils/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   base/index.rst
   dimenet_plus_plus/index.rst
   model_registry/index.rst
   schnet/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   core.models.model_name_to_local_file



Attributes
~~~~~~~~~~

.. autoapisummary::

   core.models.available_pretrained_models


.. py:data:: available_pretrained_models

   

.. py:function:: model_name_to_local_file(model_name: str, local_cache: str | pathlib.Path) -> str

   Download a pretrained checkpoint if it does not exist already

   :param model_name: the model name. See available_pretrained_checkpoints.
   :type model_name: str
   :param local_cache: path to local cache directory
   :type local_cache: str or Path

   :returns: local path to checkpoint file
   :rtype: str


