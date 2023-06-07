import inspect
import json
import os
import sys
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Generic, cast

import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import TypeVar, override

from .. import actsave
from ..trainer import Trainer as LLTrainer
from ..util import log_batch_info, skip_batch
from .config import BaseConfig
from .modules.callback import CallbackModuleMixin, CallbackRegistrarModuleMixin
from .modules.dataset import DatasetModuleMixin
from .modules.debug import DebugModuleMixin
from .modules.distributed import DistributedMixin
from .modules.finite_checks import FiniteChecksModuleMixin
from .modules.log_dir import LogDirMixin
from .modules.log_epoch import LogEpochMixin
from .modules.logger import LoggerModuleMixin
from .modules.lr_monitor import LRMonitorMixin
from .modules.optimizer import OptimizerModuleMixin
from .modules.parameter_hooks import ParameterHookModuleMixin
from .modules.rlp_sanity_checks import RLPSanityCheckModuleMixin
from .modules.shared_parameters import SharedParametersModuleMixin
from .modules.wandb import WandbWrapperMixin

log = getLogger(__name__)

THparams = TypeVar("THparams", bound=BaseConfig, infer_variance=True)


class Base(DebugModuleMixin, Generic[THparams]):
    @torch.jit.unused
    @property
    def config(self) -> THparams:
        return self.hparams

    @property
    def debug(self) -> bool:
        if torch.jit.is_scripting():
            return False
        return self.config.debug

    @property
    def dev(self) -> bool:
        if torch.jit.is_scripting():
            return False
        return self.config.debug

    @override
    def __init__(self, hparams: THparams):
        super().__init__()

        hparams.validate()

        if not hasattr(self, "hparams"):
            self.hparams = hparams


class DebugFlagCallback(Callback):
    """
    Sets the debug flag to true in the following circumstances:
    - fast_dev_run is enabled
    - sanity check is running
    """

    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        if not getattr(trainer, "fast_dev_run", False):
            return

        hparams = cast(BaseConfig, pl_module.hparams)
        if not hparams.debug:
            log.critical("Fast dev run detected, setting debug flag to True.")
        hparams.debug = True

    @override
    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule):
        hparams = cast(BaseConfig, pl_module.hparams)
        self._debug = hparams.debug
        if not self._debug:
            log.critical("Enabling debug flag during sanity check routine.")
        hparams.debug = True

    @override
    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule):
        hparams = cast(BaseConfig, pl_module.hparams)
        if not self._debug:
            log.critical("Sanity check routine complete, disabling debug flag.")
        hparams.debug = self._debug


def _slurm_session_info():
    try:
        from submitit import JobEnvironment

        job = JobEnvironment()
        if not job.activated():
            return {}

        return {
            "hostname": job.hostname,
            "hostnames": job.hostnames,
            "job_id": job.job_id,
            "raw_job_id": job.raw_job_id,
            "array_job_id": job.array_job_id,
            "array_task_id": job.array_task_id,
            "num_tasks": job.num_tasks,
            "num_nodes": job.num_nodes,
            "node": job.node,
            "global_rank": job.global_rank,
            "local_rank": job.local_rank,
        }
    except (ImportError, RuntimeError):
        return {}


def _cls_info(cls: type):
    name = cls.__name__
    module = cls.__module__
    full_name = f"{cls.__module__}.{cls.__qualname__}"

    file_path = inspect.getfile(cls)
    source_file_path = inspect.getsourcefile(cls)

    return {
        "name": name,
        "module": module,
        "full_name": full_name,
        "file_path": file_path,
        "source_file_path": source_file_path,
    }


class LightningModuleBase(
    LogDirMixin,
    WandbWrapperMixin,
    OptimizerModuleMixin,
    RLPSanityCheckModuleMixin,
    LogEpochMixin,
    LoggerModuleMixin,
    LRMonitorMixin,
    DatasetModuleMixin,
    FiniteChecksModuleMixin,
    SharedParametersModuleMixin,
    ParameterHookModuleMixin,
    DistributedMixin,
    CallbackModuleMixin,
    Base[THparams],
    LightningModule,
    ABC,
    Generic[THparams],
):
    hparams: THparams
    hparams_initial: THparams

    @classmethod
    @abstractmethod
    def config_cls(cls) -> type[THparams]:
        ...

    @classmethod
    def _update_environment(cls, hparams: THparams):
        hparams.environment.cwd = os.getcwd()
        hparams.environment.python_executable = sys.executable
        hparams.environment.python_path = sys.path
        hparams.environment.python_version = sys.version
        hparams.environment.config = _cls_info(cls.config_cls())
        hparams.environment.model = _cls_info(cls)
        hparams.environment.slurm = _slurm_session_info()
        hparams.environment.log_dir = str(
            LLTrainer.ll_default_root_dir(hparams).absolute()
        )
        hparams.environment.seed = (
            int(seed_str) if (seed_str := os.environ.get("PL_GLOBAL_SEED")) else None
        )
        hparams.environment.seed_workers = (
            bool(int(seed_everything))
            if (seed_everything := os.environ.get("PL_SEED_WORKERS"))
            else None
        )
        hparams.environment.sweep_id = os.environ.get("LL_WANDB_SWEEP_ID")
        hparams.environment.sweep_config = (
            json.loads(config_json)
            if (config_json := os.environ.get("LL_WANDB_SWEEP_CONFIG")) is not None
            else None
        )

    @override
    def __init__(self, hparams: THparams):
        self._update_environment(hparams)

        super().__init__(hparams)

        self.save_hyperparameters(hparams)

        self.register_callback(lambda: DebugFlagCallback())

        actsave.wrap_lightning_module(self)

        if self.config.trainer.log_batch_info_on_error:
            log_batch_info.wrap_lightning_module(self)

        if self.config.trainer.supports_skip_batch_exception:
            skip_batch.wrap_lightning_module(self)

    def zero_loss(self):
        """
        Returns a loss tensor with the value 0.
        It multiples each weight by 0 and returns the sum, so we don't run into issues with ununsed parameters in DDP.
        """
        loss = sum((0.0 * v).sum() for v in self.parameters())
        loss = cast(torch.Tensor, loss)
        return loss

    def skip_batch_training_step(self, *args: Any, **kwargs: Any):
        """
        This function gets called when a `SkipBatch` exception is raised during any point in training.
        If `automatic_optimization` is enabled, it should return a loss tensor that will be used for the backward pass. By default, it returns a zero loss tensor.
        If `automatic_optimization` is disabled, this function needs to be implemented and should handle the backward pass itself.
        """
        if not self.automatic_optimization:
            raise NotImplementedError(
                "To use `SkipBatch` with manual optimization, you must implement `skip_batch_training_step`."
            )

        loss = self.zero_loss()
        return loss

    @property
    def datamodule(self):
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return None

        if not isinstance(datamodule, LightningDataModuleBase):
            raise TypeError(
                f"datamodule must be a LightningDataModuleBase: {type(datamodule)}"
            )

        datamodule = cast(LightningDataModuleBase[THparams], datamodule)
        return datamodule

    @abstractmethod
    @override
    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        ...

    @abstractmethod
    @override
    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        ...


class LightningDataModuleBase(
    LogDirMixin,
    DatasetModuleMixin,
    CallbackRegistrarModuleMixin,
    Base[THparams],
    LightningDataModule,
    ABC,
    Generic[THparams],
):
    hparams: THparams
    hparams_initial: THparams

    @classmethod
    def _update_environment(cls, hparams: THparams):
        hparams.environment.data = _cls_info(cls)

    @override
    def __init__(self, hparams: THparams):
        self._update_environment(hparams)
        super().__init__(hparams)

        self.save_hyperparameters(hparams)

    @property
    def lightning_module(self):
        if not self.trainer:
            raise ValueError("Trainer has not been set.")

        module = self.trainer.lightning_module
        if not isinstance(module, LightningModuleBase):
            raise ValueError(
                f"Trainer's lightning_module is not a LightningModuleBase: {type(module)}"
            )

        module = cast(LightningModuleBase[THparams], module)
        return module

    @property
    def device(self):
        return self.lightning_module.device
