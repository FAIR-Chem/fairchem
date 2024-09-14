core.modules.scaling.fit
========================

.. py:module:: core.modules.scaling.fit


Attributes
----------

.. autoapisummary::

   core.modules.scaling.fit.parser


Functions
---------

.. autoapisummary::

   core.modules.scaling.fit._prefilled_input
   core.modules.scaling.fit._train_batch
   core.modules.scaling.fit.compute_scaling_factors


Module Contents
---------------

.. py:function:: _prefilled_input(prompt: str, prefill: str = '') -> str

.. py:function:: _train_batch(trainer: fairchem.core.trainers.base_trainer.BaseTrainer, batch) -> None

.. py:function:: compute_scaling_factors(config, num_batches: int = 16) -> None

.. py:data:: parser

