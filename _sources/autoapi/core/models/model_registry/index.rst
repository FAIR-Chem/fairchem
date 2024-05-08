:py:mod:`core.models.model_registry`
====================================

.. py:module:: core.models.model_registry

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   core.models.model_registry.model_name_to_local_file



Attributes
~~~~~~~~~~

.. autoapisummary::

   core.models.model_registry.MODEL_REGISTRY
   core.models.model_registry.available_pretrained_models


.. py:data:: MODEL_REGISTRY

   

.. py:data:: available_pretrained_models

   

.. py:function:: model_name_to_local_file(model_name: str, local_cache: str | pathlib.Path) -> str

   Download a pretrained checkpoint if it does not exist already

   :param model_name: the model name. See available_pretrained_checkpoints.
   :type model_name: str
   :param local_cache: path to local cache directory
   :type local_cache: str or Path

   :returns: local path to checkpoint file
   :rtype: str


