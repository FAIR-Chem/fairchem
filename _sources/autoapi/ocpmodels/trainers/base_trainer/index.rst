:py:mod:`ocpmodels.trainers.base_trainer`
=========================================

.. py:module:: ocpmodels.trainers.base_trainer

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.trainers.base_trainer.BaseTrainer




.. py:class:: BaseTrainer(task, model, outputs, dataset, optimizer, loss_fns, eval_metrics, identifier: str, timestamp_id: str | None = None, run_dir: str | None = None, is_debug: bool = False, print_every: int = 100, seed: int | None = None, logger: str = 'wandb', local_rank: int = 0, amp: bool = False, cpu: bool = False, name: str = 'ocp', slurm=None, noddp: bool = False)


   Bases: :py:obj:`abc.ABC`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: _unwrapped_model


   .. py:method:: train(disable_eval_tqdm: bool = False) -> None
      :abstractmethod:

      Run model training iterations.


   .. py:method:: _get_timestamp(device: torch.device, suffix: str | None) -> str
      :staticmethod:


   .. py:method:: load() -> None


   .. py:method:: set_seed(seed) -> None


   .. py:method:: load_seed_from_config() -> None


   .. py:method:: load_logger() -> None


   .. py:method:: get_sampler(dataset, batch_size: int, shuffle: bool) -> ocpmodels.common.data_parallel.BalancedBatchSampler


   .. py:method:: get_dataloader(dataset, sampler) -> torch.utils.data.DataLoader


   .. py:method:: load_datasets() -> None


   .. py:method:: load_task()


   .. py:method:: load_model() -> None


   .. py:method:: load_checkpoint(checkpoint_path: str, checkpoint: dict | None = None) -> None


   .. py:method:: load_loss() -> None


   .. py:method:: load_optimizer() -> None


   .. py:method:: load_extras() -> None


   .. py:method:: save(metrics=None, checkpoint_file: str = 'checkpoint.pt', training_state: bool = True) -> str | None


   .. py:method:: update_best(primary_metric, val_metrics, disable_eval_tqdm: bool = True) -> None


   .. py:method:: validate(split: str = 'val', disable_tqdm: bool = False)


   .. py:method:: _backward(loss) -> None


   .. py:method:: save_results(predictions, results_file: str | None, keys=None) -> None



