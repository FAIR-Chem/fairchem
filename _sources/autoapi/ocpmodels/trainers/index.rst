:py:mod:`ocpmodels.trainers`
============================

.. py:module:: ocpmodels.trainers


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   base_trainer/index.rst
   ocp_trainer/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.trainers.BaseTrainer
   ocpmodels.trainers.OCPTrainer




.. py:class:: BaseTrainer(task, model, outputs, dataset, optimizer, loss_fns, eval_metrics, identifier: str, timestamp_id: Optional[str] = None, run_dir: Optional[str] = None, is_debug: bool = False, print_every: int = 100, seed: Optional[int] = None, logger: str = 'tensorboard', local_rank: int = 0, amp: bool = False, cpu: bool = False, name: str = 'ocp', slurm={}, noddp: bool = False)


   Bases: :py:obj:`abc.ABC`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: _unwrapped_model


   .. py:method:: _get_timestamp(device: torch.device, suffix: Optional[str]) -> str
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


   .. py:method:: load_checkpoint(checkpoint_path: str, checkpoint: Dict = {}) -> None


   .. py:method:: load_loss() -> None


   .. py:method:: load_optimizer() -> None


   .. py:method:: load_extras() -> None


   .. py:method:: save(metrics=None, checkpoint_file: str = 'checkpoint.pt', training_state: bool = True) -> Optional[str]


   .. py:method:: update_best(primary_metric, val_metrics, disable_eval_tqdm: bool = True) -> None


   .. py:method:: validate(split: str = 'val', disable_tqdm: bool = False)


   .. py:method:: _backward(loss) -> None


   .. py:method:: save_results(predictions, results_file: Optional[str], keys=None) -> None



.. py:class:: OCPTrainer(task, model, outputs, dataset, optimizer, loss_fns, eval_metrics, identifier, timestamp_id=None, run_dir=None, is_debug=False, print_every=100, seed=None, logger='tensorboard', local_rank=0, amp=False, cpu=False, slurm={}, noddp=False, name='ocp')


   Bases: :py:obj:`ocpmodels.trainers.base_trainer.BaseTrainer`

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
                  (default: :obj:`tensorboard`)
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


   .. py:method:: _forward(batch)


   .. py:method:: _compute_loss(out, batch)


   .. py:method:: _compute_metrics(out, batch, evaluator, metrics={})


   .. py:method:: predict(data_loader, per_image: bool = True, results_file: Optional[str] = None, disable_tqdm: bool = False)


   .. py:method:: run_relaxations(split='val')



