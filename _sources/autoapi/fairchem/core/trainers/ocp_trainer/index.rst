:py:mod:`fairchem.core.trainers.ocp_trainer`
============================================

.. py:module:: fairchem.core.trainers.ocp_trainer

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.core.trainers.ocp_trainer.OCPTrainer




.. py:class:: OCPTrainer(task, model, outputs, dataset, optimizer, loss_fns, eval_metrics, identifier, timestamp_id=None, run_dir=None, is_debug=False, print_every=100, seed=None, logger='wandb', local_rank=0, amp=False, cpu=False, slurm=None, noddp=False, name='ocp')


   Bases: :py:obj:`fairchem.core.trainers.base_trainer.BaseTrainer`

   Trainer class for the Structure to Energy & Force (S2EF) and Initial State to
   Relaxed State (IS2RS) tasks.

   .. note::

       Examples of configurations for task, model, dataset and optimizer
       can be found in `configs/ocp_s2ef <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_
       and `configs/ocp_is2rs <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2rs/>`_.

   :param task: Task configuration.
   :type task: dict
   :param model: Model configuration.
   :type model: dict
   :param outputs: Output property configuration.
   :type outputs: dict
   :param dataset: Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
   :type dataset: dict
   :param optimizer: Optimizer configuration.
   :type optimizer: dict
   :param loss_fns: Loss function configuration.
   :type loss_fns: dict
   :param eval_metrics: Evaluation metrics configuration.
   :type eval_metrics: dict
   :param identifier: Experiment identifier that is appended to log directory.
   :type identifier: str
   :param run_dir: Path to the run directory where logs are to be saved.
                   (default: :obj:`None`)
   :type run_dir: str, optional
   :param is_debug: Run in debug mode.
                    (default: :obj:`False`)
   :type is_debug: bool, optional
   :param print_every: Frequency of printing logs.
                       (default: :obj:`100`)
   :type print_every: int, optional
   :param seed: Random number seed.
                (default: :obj:`None`)
   :type seed: int, optional
   :param logger: Type of logger to be used.
                  (default: :obj:`wandb`)
   :type logger: str, optional
   :param local_rank: Local rank of the process, only applicable for distributed training.
                      (default: :obj:`0`)
   :type local_rank: int, optional
   :param amp: Run using automatic mixed precision.
               (default: :obj:`False`)
   :type amp: bool, optional
   :param slurm: Slurm configuration. Currently just for keeping track.
                 (default: :obj:`{}`)
   :type slurm: dict
   :param noddp: Run model without DDP.
   :type noddp: bool, optional

   .. py:method:: train(disable_eval_tqdm: bool = False) -> None

      Run model training iterations.


   .. py:method:: _forward(batch)


   .. py:method:: _compute_loss(out, batch)


   .. py:method:: _compute_metrics(out, batch, evaluator, metrics=None)


   .. py:method:: predict(data_loader, per_image: bool = True, results_file: str | None = None, disable_tqdm: bool = False)


   .. py:method:: run_relaxations(split='val')



