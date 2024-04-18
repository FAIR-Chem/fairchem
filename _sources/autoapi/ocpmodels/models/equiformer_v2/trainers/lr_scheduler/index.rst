:py:mod:`ocpmodels.models.equiformer_v2.trainers.lr_scheduler`
==============================================================

.. py:module:: ocpmodels.models.equiformer_v2.trainers.lr_scheduler


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.equiformer_v2.trainers.lr_scheduler.CosineLRLambda
   ocpmodels.models.equiformer_v2.trainers.lr_scheduler.MultistepLRLambda
   ocpmodels.models.equiformer_v2.trainers.lr_scheduler.LRScheduler



Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.models.equiformer_v2.trainers.lr_scheduler.multiply
   ocpmodels.models.equiformer_v2.trainers.lr_scheduler.cosine_lr_lambda
   ocpmodels.models.equiformer_v2.trainers.lr_scheduler.multistep_lr_lambda



.. py:function:: multiply(obj, num)


.. py:function:: cosine_lr_lambda(current_step: int, scheduler_params)


.. py:class:: CosineLRLambda(scheduler_params)


   .. py:method:: __call__(current_step: int)



.. py:function:: multistep_lr_lambda(current_step: int, scheduler_params) -> float


.. py:class:: MultistepLRLambda(scheduler_params)


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

   .. py:method:: step(metrics=None, epoch=None)


   .. py:method:: filter_kwargs(config)


   .. py:method:: get_lr() -> Optional[float]



