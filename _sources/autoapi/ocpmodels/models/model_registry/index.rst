:py:mod:`ocpmodels.models.model_registry`
=========================================

.. py:module:: ocpmodels.models.model_registry


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.models.model_registry.model_name_to_local_file



Attributes
~~~~~~~~~~

.. autoapisummary::

   ocpmodels.models.model_registry.MODEL_REGISTRY
   ocpmodels.models.model_registry.available_pretrained_models


.. py:data:: MODEL_REGISTRY

   

.. py:data:: available_pretrained_models

   

.. py:function:: model_name_to_local_file(model_name: str, local_cache: str | pathlib.Path) -> str

   Download a pretrained checkpoint if it does not exist already

   :param model_name: the model name. See available_pretrained_checkpoints.
   :type model_name: str
   :param local_cache:
   :type local_cache: str

   Returns:



