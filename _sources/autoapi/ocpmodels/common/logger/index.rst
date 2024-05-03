:py:mod:`ocpmodels.common.logger`
=================================

.. py:module:: ocpmodels.common.logger

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.common.logger.Logger
   ocpmodels.common.logger.WandBLogger
   ocpmodels.common.logger.TensorboardLogger




.. py:class:: Logger(config)


   Bases: :py:obj:`abc.ABC`

   Generic class to interface with various logging modules, e.g. wandb,
   tensorboard, etc.

   .. py:method:: watch(model)
      :abstractmethod:

      Monitor parameters and gradients.


   .. py:method:: log(update_dict, step: int, split: str = '')

      Log some values.


   .. py:method:: log_plots(plots) -> None
      :abstractmethod:


   .. py:method:: mark_preempting() -> None
      :abstractmethod:



.. py:class:: WandBLogger(config)


   Bases: :py:obj:`Logger`

   Generic class to interface with various logging modules, e.g. wandb,
   tensorboard, etc.

   .. py:method:: watch(model) -> None

      Monitor parameters and gradients.


   .. py:method:: log(update_dict, step: int, split: str = '') -> None

      Log some values.


   .. py:method:: log_plots(plots, caption: str = '') -> None


   .. py:method:: mark_preempting() -> None



.. py:class:: TensorboardLogger(config)


   Bases: :py:obj:`Logger`

   Generic class to interface with various logging modules, e.g. wandb,
   tensorboard, etc.

   .. py:method:: watch(model) -> bool

      Monitor parameters and gradients.


   .. py:method:: log(update_dict, step: int, split: str = '')

      Log some values.


   .. py:method:: mark_preempting() -> None


   .. py:method:: log_plots(plots) -> None



