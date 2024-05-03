:py:mod:`ocpmodels.common.registry`
===================================

.. py:module:: ocpmodels.common.registry

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.

   # Copyright (c) Meta, Inc. and its affiliates.
   # Borrowed from https://github.com/facebookresearch/pythia/blob/master/pythia/common/registry.py.

   Registry is central source of truth. Inspired from Redux's concept of
   global store, Registry maintains mappings of various information to unique
   keys. Special functions in registry can be used as decorators to register
   different kind of classes.

   Import the global registry object using

   ``from ocpmodels.common.registry import registry``

   Various decorators for registry different kind of classes with unique keys

   - Register a model: ``@registry.register_model``



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.common.registry.Registry



Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.common.registry._get_absolute_mapping



Attributes
~~~~~~~~~~

.. autoapisummary::

   ocpmodels.common.registry.R
   ocpmodels.common.registry.NestedDict
   ocpmodels.common.registry.registry


.. py:data:: R

   

.. py:data:: NestedDict

   

.. py:function:: _get_absolute_mapping(name: str)


.. py:class:: Registry


   Class for registry object which acts as central source of truth.

   .. py:attribute:: mapping
      :type: ClassVar[NestedDict]

      

   .. py:method:: register_task(name: str)
      :classmethod:

      Register a new task to registry with key 'name'
      :param name: Key with which the task will be registered.

      Usage::
          from ocpmodels.common.registry import registry
          from ocpmodels.tasks import BaseTask
          @registry.register_task("train")
          class TrainTask(BaseTask):
              ...


   .. py:method:: register_dataset(name: str)
      :classmethod:

      Register a dataset to registry with key 'name'

      :param name: Key with which the dataset will be registered.

      Usage::

          from ocpmodels.common.registry import registry
          from ocpmodels.datasets import BaseDataset

          @registry.register_dataset("qm9")
          class QM9(BaseDataset):
              ...


   .. py:method:: register_model(name: str)
      :classmethod:

      Register a model to registry with key 'name'

      :param name: Key with which the model will be registered.

      Usage::

          from ocpmodels.common.registry import registry
          from ocpmodels.modules.layers import CGCNNConv

          @registry.register_model("cgcnn")
          class CGCNN():
              ...


   .. py:method:: register_logger(name: str)
      :classmethod:

      Register a logger to registry with key 'name'

      :param name: Key with which the logger will be registered.

      Usage::

          from ocpmodels.common.registry import registry

          @registry.register_logger("wandb")
          class WandBLogger():
              ...


   .. py:method:: register_trainer(name: str)
      :classmethod:

      Register a trainer to registry with key 'name'

      :param name: Key with which the trainer will be registered.

      Usage::

          from ocpmodels.common.registry import registry

          @registry.register_trainer("active_discovery")
          class ActiveDiscoveryTrainer():
              ...


   .. py:method:: register(name: str, obj) -> None
      :classmethod:

      Register an item to registry with key 'name'

      :param name: Key with which the item will be registered.

      Usage::

          from ocpmodels.common.registry import registry

          registry.register("config", {})


   .. py:method:: __import_error(name: str, mapping_name: str) -> RuntimeError
      :classmethod:


   .. py:method:: get_class(name: str, mapping_name: str)
      :classmethod:


   .. py:method:: get_task_class(name: str)
      :classmethod:


   .. py:method:: get_dataset_class(name: str)
      :classmethod:


   .. py:method:: get_model_class(name: str)
      :classmethod:


   .. py:method:: get_logger_class(name: str)
      :classmethod:


   .. py:method:: get_trainer_class(name: str)
      :classmethod:


   .. py:method:: get(name: str, default=None, no_warning: bool = False)
      :classmethod:

      Get an item from registry with key 'name'

      :param name: Key whose value needs to be retrieved.
      :type name: string
      :param default: If passed and key is not in registry, default value will
                      be returned with a warning. Default: None
      :param no_warning: If passed as True, warning when key doesn't exist
                         will not be generated. Useful for cgcnn's
                         internal operations. Default: False
      :type no_warning: bool

      Usage::

          from ocpmodels.common.registry import registry

          config = registry.get("config")


   .. py:method:: unregister(name: str)
      :classmethod:

      Remove an item from registry with key 'name'

      :param name: Key which needs to be removed.

      Usage::

          from ocpmodels.common.registry import registry

          config = registry.unregister("config")



.. py:data:: registry

   

