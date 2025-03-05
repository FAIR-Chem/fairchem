core._cli_hydra
===============

.. py:module:: core._cli_hydra

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   core._cli_hydra.ALLOWED_TOP_LEVEL_KEYS
   core._cli_hydra.LOG_DIR_NAME
   core._cli_hydra.CHECKPOINT_DIR_NAME
   core._cli_hydra.RESULTS_DIR
   core._cli_hydra.CONFIG_FILE_NAME
   core._cli_hydra.PREEMPTION_STATE_DIR_NAME


Classes
-------

.. autoapisummary::

   core._cli_hydra.SchedulerType
   core._cli_hydra.DeviceType
   core._cli_hydra.SlurmConfig
   core._cli_hydra.SchedulerConfig
   core._cli_hydra.Metadata
   core._cli_hydra.JobConfig
   core._cli_hydra.Submitit


Functions
---------

.. autoapisummary::

   core._cli_hydra._set_seeds
   core._cli_hydra._set_deterministic_mode
   core._cli_hydra.map_job_config_to_dist_config
   core._cli_hydra.get_canonical_config
   core._cli_hydra.get_hydra_config_from_yaml
   core._cli_hydra.runner_wrapper
   core._cli_hydra.main


Module Contents
---------------

.. py:data:: ALLOWED_TOP_LEVEL_KEYS

.. py:data:: LOG_DIR_NAME
   :value: 'logs'


.. py:data:: CHECKPOINT_DIR_NAME
   :value: 'checkpoints'


.. py:data:: RESULTS_DIR
   :value: 'results'


.. py:data:: CONFIG_FILE_NAME
   :value: 'canonical_config.yaml'


.. py:data:: PREEMPTION_STATE_DIR_NAME
   :value: 'preemption_state'


.. py:class:: SchedulerType

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`


   str(object='') -> str
   str(bytes_or_buffer[, encoding[, errors]]) -> str

   Create a new string object from the given object. If encoding or
   errors is specified, then the object must expose a data buffer
   that will be decoded using the given encoding and error handler.
   Otherwise, returns the result of object.__str__() (if defined)
   or repr(object).
   encoding defaults to sys.getdefaultencoding().
   errors defaults to 'strict'.


   .. py:attribute:: LOCAL
      :value: 'local'



   .. py:attribute:: SLURM
      :value: 'slurm'



.. py:class:: DeviceType

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`


   str(object='') -> str
   str(bytes_or_buffer[, encoding[, errors]]) -> str

   Create a new string object from the given object. If encoding or
   errors is specified, then the object must expose a data buffer
   that will be decoded using the given encoding and error handler.
   Otherwise, returns the result of object.__str__() (if defined)
   or repr(object).
   encoding defaults to sys.getdefaultencoding().
   errors defaults to 'strict'.


   .. py:attribute:: CPU
      :value: 'cpu'



   .. py:attribute:: CUDA
      :value: 'cuda'



.. py:class:: SlurmConfig

   .. py:attribute:: mem_gb
      :type:  int
      :value: 80



   .. py:attribute:: timeout_hr
      :type:  int
      :value: 168



   .. py:attribute:: cpus_per_task
      :type:  int
      :value: 8



   .. py:attribute:: partition
      :type:  Optional[str]
      :value: None



   .. py:attribute:: qos
      :type:  Optional[str]
      :value: None



   .. py:attribute:: account
      :type:  Optional[str]
      :value: None



.. py:class:: SchedulerConfig

   .. py:attribute:: mode
      :type:  SchedulerType


   .. py:attribute:: ranks_per_node
      :type:  int
      :value: 1



   .. py:attribute:: num_nodes
      :type:  int
      :value: 1



   .. py:attribute:: slurm
      :type:  SlurmConfig


.. py:class:: Metadata

   .. py:attribute:: commit
      :type:  str


   .. py:attribute:: log_dir
      :type:  str


   .. py:attribute:: checkpoint_dir
      :type:  str


   .. py:attribute:: results_dir
      :type:  str


   .. py:attribute:: config_path
      :type:  str


   .. py:attribute:: preemption_checkpoint_dir
      :type:  str


   .. py:attribute:: cluster_name
      :type:  str


.. py:class:: JobConfig

   .. py:attribute:: run_name
      :type:  str


   .. py:attribute:: timestamp_id
      :type:  str


   .. py:attribute:: run_dir
      :type:  str


   .. py:attribute:: device_type
      :type:  DeviceType


   .. py:attribute:: debug
      :type:  bool
      :value: False



   .. py:attribute:: scheduler
      :type:  SchedulerConfig


   .. py:attribute:: logger
      :type:  Optional[dict]
      :value: None



   .. py:attribute:: seed
      :type:  int
      :value: 0



   .. py:attribute:: deterministic
      :type:  bool
      :value: False



   .. py:attribute:: runner_state_path
      :type:  Optional[str]
      :value: None



   .. py:attribute:: metadata
      :type:  Optional[Metadata]
      :value: None



   .. py:attribute:: graph_parallel_group_size
      :type:  Optional[int]
      :value: None



   .. py:method:: __post_init__() -> None


.. py:function:: _set_seeds(seed: int) -> None

.. py:function:: _set_deterministic_mode() -> None

.. py:class:: Submitit

   Bases: :py:obj:`submitit.helpers.Checkpointable`


   Derived callable classes are requeued after timeout with their current
   state dumped at checkpoint.

   __call__ method must be implemented to make your class a callable.

   .. note::

      The following implementation of the checkpoint method resubmits the full current
      state of the callable (self) with the initial argument. You may want to replace the method to
      curate the state (dump a neural network to a standard format and remove it from
      the state so that not to pickle it) and change/remove the initial parameters.


   .. py:attribute:: config
      :value: None



   .. py:attribute:: runner
      :value: None



   .. py:method:: __call__(dict_config: omegaconf.DictConfig) -> None


   .. py:method:: _init_logger() -> None


   .. py:method:: checkpoint(*args, **kwargs) -> submitit.helpers.DelayedSubmission

      Resubmits the same callable with the same arguments



.. py:function:: map_job_config_to_dist_config(job_cfg: JobConfig) -> dict

.. py:function:: get_canonical_config(config: omegaconf.DictConfig) -> omegaconf.DictConfig

.. py:function:: get_hydra_config_from_yaml(config_yml: str, overrides_args: list[str]) -> omegaconf.DictConfig

.. py:function:: runner_wrapper(config: omegaconf.DictConfig)

.. py:function:: main(args: argparse.Namespace | None = None, override_args: list[str] | None = None)

