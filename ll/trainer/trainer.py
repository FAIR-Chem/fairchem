from collections import abc
from contextlib import ExitStack, contextmanager
from functools import wraps
from logging import getLogger
from types import NoneType
from typing import Any, Callable

from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT
from lightning_fabric.utilities.types import _PATH
from typing_extensions import override

from ..model.config import BaseConfig
from ..util import seed
from ..util.environment import set_additional_env_vars
from ..util.typing_utils import copy_method_with_param
from .logging import (
    default_root_dir,
    finalize_loggers,
    loggers_from_config,
    validate_logger,
)

log = getLogger(__name__)


class Trainer(LightningTrainer):
    _finalizers: list[Callable[[], None]] = []

    def finalize(self):
        """
        Call this method to clean up after training.
        """
        finalize_loggers(self.loggers)

    @staticmethod
    def ll_default_root_dir(
        config: BaseConfig,
        *,
        create_symlinks: bool = True,
        logs_dirname: str = "lightning_logs",
    ):
        return default_root_dir(
            config,
            create_symlinks=create_symlinks,
            logs_dirname=logs_dirname,
        )

    @classmethod
    @contextmanager
    def context(cls, config: BaseConfig):
        with ExitStack() as stack:
            cls._finalizers.clear()
            if config.trainer.seed is not None:
                stack.enter_context(
                    seed.seed_context(
                        config.trainer.seed, workers=config.trainer.seed_workers
                    )
                )

            additional_nccl_env_vars: dict[str, str] = {}
            if config.trainer.set_nccl_optimal_params:
                # We need to set these env vars before the NCCL library is loaded.
                # Reportedly, the training performance can be improved quite a bit (see).
                # Details on all available env vars: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
                additional_nccl_env_vars["NCCL_NSOCKS_PERTHREAD"] = "4"
                additional_nccl_env_vars["NCCL_SOCKET_NTHREADS"] = "2"
            stack.enter_context(
                set_additional_env_vars(
                    config.trainer.additional_env_vars | additional_nccl_env_vars
                )
            )

            try:
                yield
            finally:
                n_finalizers = 0
                for finalizer in reversed(cls._finalizers):
                    finalizer()
                    n_finalizers += 1

                cls._finalizers.clear()
                log.critical(
                    f"Ran {n_finalizers} finalizers for {cls.__name__} cleanup."
                )

    @classmethod
    def _update_kwargs(cls, config: BaseConfig, kwargs: dict[str, Any]):
        kwargs_ = {
            "accelerator": config.trainer.accelerator,
            "strategy": config.trainer.strategy,
            "devices": config.trainer.devices,
            "num_nodes": config.trainer.num_nodes,
            "precision": config.trainer.precision,
            "logger": config.trainer.logger,
            "fast_dev_run": config.trainer.fast_dev_run,
            "max_epochs": config.trainer.max_epochs,
            "min_epochs": config.trainer.min_epochs,
            "max_steps": config.trainer.max_steps,
            "min_steps": config.trainer.min_steps,
            "max_time": config.trainer.max_time,
            "limit_train_batches": config.trainer.limit_train_batches,
            "limit_val_batches": config.trainer.limit_val_batches,
            "limit_test_batches": config.trainer.limit_test_batches,
            "limit_predict_batches": config.trainer.limit_predict_batches,
            "overfit_batches": config.trainer.overfit_batches,
            "val_check_interval": config.trainer.val_check_interval,
            "check_val_every_n_epoch": config.trainer.check_val_every_n_epoch,
            "num_sanity_val_steps": config.trainer.num_sanity_val_steps,
            "log_every_n_steps": config.trainer.log_every_n_steps,
            "enable_checkpointing": config.trainer.enable_checkpointing,
            "enable_progress_bar": config.trainer.enable_progress_bar,
            "enable_model_summary": config.trainer.enable_model_summary,
            "accumulate_grad_batches": config.trainer.accumulate_grad_batches,
            "gradient_clip_val": config.trainer.gradient_clip_val,
            "gradient_clip_algorithm": config.trainer.gradient_clip_algorithm,
            "deterministic": config.trainer.deterministic,
            "benchmark": config.trainer.benchmark,
            "inference_mode": config.trainer.inference_mode,
            "use_distributed_sampler": config.trainer.use_distributed_sampler,
            "profiler": config.trainer.profiler,
            "detect_anomaly": config.trainer.detect_anomaly,
            "barebones": config.trainer.barebones,
            "plugins": config.trainer.plugins,
            "sync_batchnorm": config.trainer.sync_batchnorm,
            "reload_dataloaders_every_n_epochs": config.trainer.reload_dataloaders_every_n_epochs,
        }
        kwargs_.update(kwargs)

        if plugins := kwargs.get("plugins"):
            if not isinstance(plugins, list):
                plugins = [plugins]

            existing_plugins = kwargs_.pop("plugins")
            if not existing_plugins:
                existing_plugins = []
            if not isinstance(existing_plugins, list):
                existing_plugins = [existing_plugins]

            existing_plugins.extend(plugins)
            kwargs_["plugins"] = existing_plugins

        if config.trainer.logger is False:
            log.critical(f"Disabling logger because {config.trainer.logger=}.")
            kwargs_["logger"] = False
        elif kwargs_.get("logger") is False:
            log.critical(f"Disabling logger because {kwargs_.get('logger')=}.")

        if (
            existing_loggers := kwargs_.get("logger")
        ) is not False and config.trainer.auto_set_loggers:
            if int(config.trainer.fast_dev_run) > 0:
                log.critical("Disabling loggers because fast_dev_run is enabled.")
            else:
                loggers = loggers_from_config(config)
                if existing_loggers is not None and not isinstance(
                    existing_loggers, bool
                ):
                    if not isinstance(existing_loggers, list):
                        existing_loggers = [existing_loggers]
                    loggers.extend(existing_loggers)

                kwargs_["logger"] = loggers

        if kwargs_.get("num_nodes") == "auto":
            # when num_nodes is auto, we need to detect the number of nodes
            # when on slurm, this would be the number of SLURM nodes allocated
            if SLURMEnvironment.detect():
                from submitit import JobEnvironment

                job = JobEnvironment()
                if not job.activated():
                    raise ValueError(
                        "SLURMEnvironment detected through PL but not submitit. This is a bug."
                    )

                kwargs_["num_nodes"] = job.num_nodes
                log.critical(
                    f"Setting num_nodes to {job.num_nodes} (detected through submitit)."
                )
            # otherweise, we assume 1 node
            else:
                kwargs_["num_nodes"] = 1
                log.critical(f"Setting num_nodes to 1 (no SLURM detected).")

        if config.trainer.auto_set_default_root_dir:
            if config.trainer.default_root_dir:
                log.warning(
                    "You have set both `config.trainer.default_root_dir` and `config.trainer.auto_set_default_root_dir`. "
                    "The latter will be ignored."
                )
                kwargs_["default_root_dir"] = config.trainer.default_root_dir
            else:
                kwargs_["default_root_dir"] = config.trainer.default_root_dir = str(
                    cls.ll_default_root_dir(config).absolute()
                )

        kwargs_.update(config.trainer.additional_trainer_kwargs)
        return kwargs_

    @override
    @copy_method_with_param(
        LightningTrainer.__init__,
        param_type=BaseConfig,
        return_type=NoneType,
    )
    def __init__(self, config: BaseConfig, *args, **kwargs):
        self._ll_config = config
        kwargs = self._update_kwargs(config, kwargs)
        log.critical(f"LightningTrainer.__init__ with {args=} and {kwargs=}.")
        super().__init__(*args, **kwargs)

        if config.trainer.enable_logger_validation:
            for logger in self.loggers:
                validate_logger(logger, config.id)

        if config.trainer.patch_hpc_checkpoint_connector:
            self._patch_checkpoint_connector()
        if config.trainer.checkpoint_last_by_default:
            self._patch_checkpoint_last_by_default()
        if config.trainer.auto_add_trainer_finalizer:
            type(self)._finalizers.append(self.finalize)

    def _patch_checkpoint_last_by_default(self):
        """
        Patch the default ModelCheckpoint callback to save the last checkpoint by default.
        """
        enable_checkpointing = (
            True
            if self._ll_config.trainer.enable_checkpointing is None
            else self._ll_config.trainer.enable_checkpointing
        )
        if not enable_checkpointing:
            return

        if not (callbacks := getattr(self, "callbacks", None)) or not isinstance(
            callbacks, abc.Iterable
        ):
            return

        if (
            model_ckpt := next(
                (c for c in callbacks if isinstance(c, ModelCheckpoint)), None
            )
        ) is None:
            return

        log.critical(f"Setting {model_ckpt.__class__.__name__}.save_last=True.")
        model_ckpt.save_last = True
        # hacky: call the `__validate_init_configuration` method to ensure that the `save_last` parameter is valid.
        # model_ckpt.__validate_init_configuration() <- this doesn't work because it's a private method
        if (
            validate_init_configuration := getattr(
                model_ckpt,
                f"_{model_ckpt.__class__.__name__}__validate_init_configuration",
                None,
            )
        ) is not None and callable(validate_init_configuration):
            validate_init_configuration()
        else:
            log.warning(
                f"Failed to find {model_ckpt.__class__.__name__}.__validate_init_configuration. "
                "This means that we cannot validate the `save_last` parameter for ModelCheckpoint."
            )

    def _patch_checkpoint_connector(self):
        """
        Patch the checkpoint connector to ignore the checkpoint path if we are
        running on SLURM and the hpc checkpoint exists.
        This is to handle the scenario where we start training a model with some checkpoint,
        (e.g., `trainer.fit(model, ckpt_path="some/path")`), but then that job gets preempted
        by SLURM. When the job is restarted, we want to resume from the hpc checkpoint, not
        the checkpoint path.
        """
        prev_set_ckpt_path = self._checkpoint_connector._parse_ckpt_path.__func__

        trainer = self

        @wraps(prev_set_ckpt_path)
        def _parse_ckpt_path(
            self, state_fn, ckpt_path, model_provided, model_connected
        ):
            nonlocal trainer
            if ckpt_path is None and trainer._ll_config.trainer.default_ckpt_path:
                log.critical(
                    f"No `ckpt_path` provided, using default {trainer._ll_config.trainer.default_ckpt_path}."
                )
                ckpt_path = trainer._ll_config.trainer.default_ckpt_path

            if (
                ckpt_path is not None
                and SLURMEnvironment.detect()
                and (hpc_path := self._hpc_resume_path) is not None
            ):
                log.critical(
                    f"SLURM hpc checkpoint exists at {hpc_path}, ignoring {ckpt_path}"
                )
                ckpt_path = "hpc"

            return prev_set_ckpt_path(
                self, state_fn, ckpt_path, model_provided, model_connected
            )

        self._checkpoint_connector._parse_ckpt_path = _parse_ckpt_path.__get__(
            self._checkpoint_connector
        )

    @override
    def _run(
        self,
        model: LightningModule,
        ckpt_path: _PATH | None = None,
    ) -> _EVALUATE_OUTPUT | _PREDICT_OUTPUT | None:
        """
        Lightning doesn't support gradient clipping with manual optimization.
        We patch the `Trainer._run` method to disable gradient clipping if
        `model.automatic_optimization` is False.
        """
        if not model.automatic_optimization:
            if self.gradient_clip_val:
                log.critical(
                    f"Disabling gradient clipping because {model.__class__.__name__}.automatic_optimization is False."
                    " If you want to use gradient clipping with manual optimization, you can use the values in "
                    "`config.trainer.gradient_clip_val` and `config.trainer.gradient_clip_algorithm`."
                )
            self.gradient_clip_val = None
            self.gradient_clip_algorithm = None

        return super()._run(model, ckpt_path)
