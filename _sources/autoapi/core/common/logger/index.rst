core.common.logger
==================

.. py:module:: core.common.logger

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.common.logger.Logger
   core.common.logger.WandBLogger
   core.common.logger.TensorboardLogger
   core.common.logger.WandBSingletonLogger


Module Contents
---------------

.. py:class:: Logger(config)

   Bases: :py:obj:`abc.ABC`


   Generic class to interface with various logging modules, e.g. wandb,
   tensorboard, etc.


   .. py:attribute:: config


   .. py:method:: watch(model, log_freq: int = 1000)
      :abstractmethod:


      Monitor parameters and gradients.



   .. py:method:: log(update_dict, step: int, split: str = '')

      Log some values.



   .. py:method:: log_plots(plots) -> None
      :abstractmethod:



   .. py:method:: mark_preempting() -> None
      :abstractmethod:



   .. py:method:: log_summary(summary_dict: dict[str, Any]) -> None
      :abstractmethod:



   .. py:method:: log_artifact(name: str, type: str, file_location: str) -> None
      :abstractmethod:



.. py:class:: WandBLogger(config)

   Bases: :py:obj:`Logger`


   Generic class to interface with various logging modules, e.g. wandb,
   tensorboard, etc.


   .. py:attribute:: project


   .. py:attribute:: entity


   .. py:attribute:: group


   .. py:method:: watch(model, log_freq: int = 1000) -> None

      Monitor parameters and gradients.



   .. py:method:: log(update_dict, step: int, split: str = '') -> None

      Log some values.



   .. py:method:: log_plots(plots, caption: str = '') -> None


   .. py:method:: log_summary(summary_dict: dict[str, Any])


   .. py:method:: mark_preempting() -> None


   .. py:method:: log_artifact(name: str, type: str, file_location: str) -> None


.. py:class:: TensorboardLogger(config)

   Bases: :py:obj:`Logger`


   Generic class to interface with various logging modules, e.g. wandb,
   tensorboard, etc.


   .. py:attribute:: writer


   .. py:method:: watch(model, log_freq: int = 1000) -> bool

      Monitor parameters and gradients.



   .. py:method:: log(update_dict, step: int, split: str = '')

      Log some values.



   .. py:method:: mark_preempting() -> None


   .. py:method:: log_plots(plots) -> None


   .. py:method:: log_summary(summary_dict: dict[str, Any]) -> None


   .. py:method:: log_artifact(name: str, type: str, file_location: str) -> None


.. py:class:: WandBSingletonLogger

   Singleton version of wandb logger, this forces a single instance of the logger to be created and used from anywhere in the code (not just from the trainer).
   This will replace the original WandBLogger.

   We initialize wandb instance somewhere in the trainer/runner globally:

   WandBSingletonLogger.init_wandb(...)

   Then from anywhere in the code we can fetch the singleton instance and log to wandb,
   note this allows you to log without knowing explicitly which step you are on
   see: https://docs.wandb.ai/ref/python/log/#the-wb-step for more details

   WandBSingletonLogger.get_instance().log({"some_value": value}, commit=False)


   .. py:attribute:: _instance
      :value: None



   .. py:method:: init_wandb(config: dict, run_id: str, run_name: str, log_dir: str, project: str, entity: str, group: str | None = None) -> None
      :classmethod:



   .. py:method:: get_instance()
      :classmethod:



   .. py:method:: watch(model, log_freq: int = 1000) -> None


   .. py:method:: log(update_dict: dict, step: int | None = None, commit=False, split: str = '') -> None


   .. py:method:: log_plots(plots, caption: str = '') -> None


   .. py:method:: log_summary(summary_dict: dict[str, Any])


   .. py:method:: mark_preempting() -> None


   .. py:method:: log_artifact(name: str, type: str, file_location: str) -> None


