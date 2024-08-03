core.models.equiformer_v2.trainers.lr_scheduler
===============================================

.. py:module:: core.models.equiformer_v2.trainers.lr_scheduler


Classes
-------

.. autoapisummary::

   core.models.equiformer_v2.trainers.lr_scheduler.CosineLRLambda
   core.models.equiformer_v2.trainers.lr_scheduler.MultistepLRLambda
   core.models.equiformer_v2.trainers.lr_scheduler.LRScheduler


Functions
---------

.. autoapisummary::

   core.models.equiformer_v2.trainers.lr_scheduler.multiply
   core.models.equiformer_v2.trainers.lr_scheduler.cosine_lr_lambda
   core.models.equiformer_v2.trainers.lr_scheduler.multistep_lr_lambda


Module Contents
---------------

.. py:function:: multiply(obj, num)

.. py:function:: cosine_lr_lambda(current_step: int, scheduler_params)

.. py:class:: CosineLRLambda(scheduler_params)

   .. py:attribute:: warmup_epochs


   .. py:attribute:: lr_warmup_factor


   .. py:attribute:: max_epochs


   .. py:attribute:: lr_min_factor


   .. py:method:: __call__(current_step: int)


.. py:function:: multistep_lr_lambda(current_step: int, scheduler_params) -> float

.. py:class:: MultistepLRLambda(scheduler_params)

   .. py:attribute:: warmup_epochs


   .. py:attribute:: lr_warmup_factor


   .. py:attribute:: lr_decay_epochs


   .. py:attribute:: lr_gamma


   .. py:method:: __call__(current_step: int) -> float


.. py:class:: LRScheduler(optimizer, config)

   .. rubric:: Notes

   1. scheduler.step() is called for every step for OC20 training.
   2. We use "scheduler_params" in .yml to specify scheduler parameters.
   3. For cosine learning rate, we use LambdaLR with lambda function being cosine:
       scheduler: LambdaLR
       scheduler_params:
           lambda_type: cosine
           ...
   4. Following 3., if `cosine` is used, `scheduler_params` in .yml looks like:
       scheduler: LambdaLR
       scheduler_params:
           lambda_type: cosine
           warmup_epochs: ...
           warmup_factor: ...
           lr_min_factor: ...
   5. Following 3., if `multistep` is used, `scheduler_params` in .yml looks like:
       scheduler: LambdaLR
       scheduler_params:
           lambda_type: multistep
           warmup_epochs: ...
           warmup_factor: ...
           decay_epochs: ... (list)
           decay_rate: ...

   :param optimizer: torch optim object
   :type optimizer: obj
   :param config: Optim dict from the input config
   :type config: dict


   .. py:attribute:: optimizer


   .. py:attribute:: config


   .. py:attribute:: scheduler_type


   .. py:attribute:: scheduler_params


   .. py:method:: step(metrics=None, epoch=None)


   .. py:method:: filter_kwargs(config)


   .. py:method:: get_lr() -> float | None


