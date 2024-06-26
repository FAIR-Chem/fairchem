core.modules.scheduler
======================

.. py:module:: core.modules.scheduler


Classes
-------

.. autoapisummary::

   core.modules.scheduler.LRScheduler


Module Contents
---------------

.. py:class:: LRScheduler(optimizer, config)

   Learning rate scheduler class for torch.optim learning rate schedulers

   .. rubric:: Notes

   If no learning rate scheduler is specified in the config the default
   scheduler is warmup_lr_lambda (fairchem.core.common.utils) not no scheduler,
   this is for backward-compatibility reasons. To run without a lr scheduler
   specify scheduler: "Null" in the optim section of the config.

   :param optimizer: torch optim object
   :type optimizer: obj
   :param config: Optim dict from the input config
   :type config: dict


   .. py:method:: step(metrics=None, epoch=None) -> None


   .. py:method:: filter_kwargs(config)


   .. py:method:: get_lr()


