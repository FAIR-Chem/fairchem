core.models.equiformer_v2.trainers.dens_trainer
===============================================

.. py:module:: core.models.equiformer_v2.trainers.dens_trainer

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.equiformer_v2.trainers.dens_trainer.DenoisingPosParams
   core.models.equiformer_v2.trainers.dens_trainer.DenoisingForcesTrainer


Functions
---------

.. autoapisummary::

   core.models.equiformer_v2.trainers.dens_trainer.add_gaussian_noise_to_position
   core.models.equiformer_v2.trainers.dens_trainer.add_gaussian_noise_schedule_to_position
   core.models.equiformer_v2.trainers.dens_trainer.denoising_pos_eval
   core.models.equiformer_v2.trainers.dens_trainer.compute_atomwise_denoising_pos_and_force_hybrid_loss


Module Contents
---------------

.. py:class:: DenoisingPosParams

   .. py:attribute:: prob
      :type:  float
      :value: 0.0



   .. py:attribute:: fixed_noise_std
      :type:  bool
      :value: False



   .. py:attribute:: std
      :type:  float
      :value: None



   .. py:attribute:: num_steps
      :type:  int
      :value: None



   .. py:attribute:: std_low
      :type:  float
      :value: None



   .. py:attribute:: std_high
      :type:  float
      :value: None



   .. py:attribute:: corrupt_ratio
      :type:  float
      :value: None



   .. py:attribute:: all_atoms
      :type:  bool
      :value: False



   .. py:attribute:: denoising_pos_coefficient
      :type:  float
      :value: None



.. py:function:: add_gaussian_noise_to_position(batch, std, corrupt_ratio=None, all_atoms=False)

   1.  Update `pos` in `batch`.
   2.  Add `noise_vec` to `batch`, which will serve as the target for denoising positions.
   3.  Add `denoising_pos_forward` to switch to denoising mode during training.
   4.  Add `noise_mask` for partially corrupted structures when `corrupt_ratio` is not None.
   5.  If `all_atoms` == True, we add noise to all atoms including fixed ones.
   6.  Check whether `batch` has `md`. We do not add noise to structures from MD split.


.. py:function:: add_gaussian_noise_schedule_to_position(batch, std_low, std_high, num_steps, corrupt_ratio=None, all_atoms=False)

   1.  Similar to above, update positions in batch with gaussian noise, but
       additionally, also save the sigmas the noise vectors are sampled from.
   2.  Add `noise_mask` for partially corrupted structures when `corrupt_ratio`
       is not None.
   3.  If `all_atoms` == True, we add noise to all atoms including fixed ones.
   4.  Check whether `batch` has `md`. We do not add noise to structures from MD split.


.. py:function:: denoising_pos_eval(evaluator: fairchem.core.modules.evaluator.Evaluator, prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor], denoising_targets: tuple[str], prev_metrics: dict[str, torch.Tensor] | None = None, denoising_pos_forward: bool = False)

   1.  Overwrite the original Evaluator.eval() here: https://github.com/Open-Catalyst-Project/ocp/blob/5a7738f9aa80b1a9a7e0ca15e33938b4d2557edd/ocpmodels/modules/evaluator.py#L69-L81
   2.  This is to make sure we separate forces MAE and denoising positions MAE.


.. py:function:: compute_atomwise_denoising_pos_and_force_hybrid_loss(pred, target, noise_mask, force_mult, denoising_pos_mult, mask=None)

.. py:class:: DenoisingForcesTrainer(task: dict[str, str | Any], model: dict[str, Any], outputs: dict[str, str | int], dataset: dict[str, str | float], optimizer: dict[str, str | float], loss_functions: dict[str, str | float], evaluation_metrics: dict[str, str], identifier: str, local_rank: int, timestamp_id: str | None = None, run_dir: str | None = None, is_debug: bool = False, print_every: int = 100, seed: int | None = None, logger: str = 'wandb', amp: bool = False, cpu: bool = False, name: str = 'ocp', slurm: dict | None = None, gp_gpus: int | None = None, inference_only: bool = False)

   Bases: :py:obj:`core.models.equiformer_v2.trainers.forces_trainer.EquiformerV2ForcesTrainer`


   1.  We add a denoising objective to the original S2EF task.
   2.  The denoising objective is that we take as input
       atom types, node-wise forces and 3D coordinates perturbed with Gaussian noises and then
       output energy of the original structure (3D coordinates without any perturbation) and
       the node-wise noises added to the original structure.
   3.  This should make models leverage more from training data and enable data augmentation for
       the S2EF task.
   4.  We should only modify the training part.
   5.  For normalizing the outputs of noise prediction, if we use `fixed_noise_std = True`, we use
       `std` for the normalization factor. Otherwise, we use `std_high` when `fixed_noise_std = False`.

   :param task: Task configuration.
   :type task: dict
   :param model: Model configuration.
   :type model: dict
   :param outputs: Dictionary of model output configuration.
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
   :param timestamp_id: timestamp identifier.
   :type timestamp_id: str, optional
   :param run_dir: Run directory used to save checkpoints and results.
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
   :param cpu: If True will run on CPU. Default is False, will attempt to use cuda.
   :type cpu: bool
   :param name: Trainer name.
   :type name: str
   :param slurm: Slurm configuration. Currently just for keeping track.
                 (default: :obj:`{}`)
   :type slurm: dict
   :param gp_gpus: Number of graph parallel GPUs.
   :type gp_gpus: int, optional
   :param inference_only: If true trainer will be loaded for inference only.
                          (ie datasets, optimizer, schedular, etc, will not be instantiated)
   :type inference_only: bool


   .. py:attribute:: use_denoising_pos


   .. py:attribute:: denoising_pos_params


   .. py:property:: denoising_targets


   .. py:method:: train(disable_eval_tqdm=False)


   .. py:method:: _compute_loss(out, batch)


   .. py:method:: _compute_metrics(out, batch, evaluator, metrics=None)


   .. py:method:: predict(data_loader, per_image: bool = True, results_file: str | None = None, disable_tqdm: bool = False)


