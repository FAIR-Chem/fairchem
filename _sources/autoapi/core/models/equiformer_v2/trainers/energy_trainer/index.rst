core.models.equiformer_v2.trainers.energy_trainer
=================================================

.. py:module:: core.models.equiformer_v2.trainers.energy_trainer

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.equiformer_v2.trainers.energy_trainer.EquiformerV2EnergyTrainer


Module Contents
---------------

.. py:class:: EquiformerV2EnergyTrainer(task: dict[str, str | Any], model: dict[str, Any], outputs: dict[str, str | int], dataset: dict[str, str | float], optimizer: dict[str, str | float], loss_functions: dict[str, str | float], evaluation_metrics: dict[str, str], identifier: str, local_rank: int, timestamp_id: str | None = None, run_dir: str | None = None, is_debug: bool = False, print_every: int = 100, seed: int | None = None, logger: str = 'wandb', amp: bool = False, cpu: bool = False, name: str = 'ocp', slurm=None, gp_gpus: int | None = None, inference_only: bool = False)

   Bases: :py:obj:`fairchem.core.trainers.OCPTrainer`


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
   :param loss_functions: Loss function configuration.
   :type loss_functions: dict
   :param evaluation_metrics: Evaluation metrics configuration.
   :type evaluation_metrics: dict
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
   :param amp: Run using automatic mixed precision.
               (default: :obj:`False`)
   :type amp: bool, optional
   :param slurm: Slurm configuration. Currently just for keeping track.
                 (default: :obj:`{}`)
   :type slurm: dict


   .. py:method:: load_extras()


