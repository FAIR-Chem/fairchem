import copy
import getpass
import os
import traceback
from collections import Counter
from contextlib import ExitStack
from datetime import timedelta
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import Generic, Protocol, cast, runtime_checkable

from tqdm.auto import tqdm
from typing_extensions import TypeVar, TypeVarTuple, Unpack, override

from submitit import AutoExecutor

from .model.config import BaseConfig
from .trainer import Trainer
from .util.environment import (
    remove_slurm_environment_variables,
    remove_wandb_environment_variables,
)
from .util.snapshot import snapshot_modules

log = getLogger(__name__)


TConfig = TypeVar("TConfig", bound=BaseConfig, infer_variance=True)
TArguments = TypeVarTuple("TArguments", default=Unpack[tuple[()]])


@runtime_checkable
class RunProtocol(Protocol[TConfig, Unpack[TArguments]]):
    def __call__(self, config: TConfig, *args: Unpack[TArguments]) -> None:
        ...


class Runner(Generic[TConfig, Unpack[TArguments]]):
    DEFAULT_ENV = {
        # Prevents HDF5 from locking files when opened in read-only mode.
        # This prevents issues when multiple processes try to read the same file.
        # See https://github.com/h5py/h5py/issues/1101#issuecomment-480354656
        "HDF5_USE_FILE_LOCKING": "FALSE",
    }
    SNAPSHOT_ENV_NAME = "LL_SNAPSHOT"

    @classmethod
    def active_snapshot(cls) -> Path | None:
        if (snapshot := os.environ.get(cls.SNAPSHOT_ENV_NAME)) is not None:
            return Path(snapshot)
        return None

    @override
    def __init__(
        self,
        run: RunProtocol[TConfig, Unpack[TArguments]],
        *,
        slurm_job_name: str = "ll",
        validate_config_before_run: bool = True,
    ):
        """This is the initialization function for a class that takes in a run protocol, an auto wrap run
        boolean, and a slurm job name string.

        Parameters
        ----------
        run : RunProtocol[TConfig, Unpack[TArguments]]
            `run` is an instance of a class that implements the `RunProtocol` interface. It represents the main function or entry point of the program that will be executed.
        slurm_job_name : str, optional
            The `slurm_job_name` parameter is a string that represents the name of the job when submitting it to a SLURM cluster.
        validate_config_before_run : bool, optional
            The `validate_config_before_run` parameter is a boolean that represents whether or not to validate the configuration before running the program.
        """

        super().__init__()

        self._run = run
        self.slurm_job_name = slurm_job_name
        self.validate_config_before_run = validate_config_before_run

    @property
    def run(self) -> RunProtocol[TConfig, Unpack[TArguments]]:
        run = self._run

        @wraps(run)
        def wrapped_run(config: TConfig, *args: Unpack[TArguments]) -> None:
            nonlocal self

            # If `validate_config_before_run`, we validate the configuration before running the program.
            if self.validate_config_before_run:
                config.validate()

            with ExitStack() as stack:
                nonlocal run

                if config.trainer.auto_wrap_trainer:
                    stack.enter_context(Trainer.context(config))
                    log.critical("Auto-wrapping run in Trainer context")

                return run(config, *args)

        return wrapped_run

    def update_config(self, config: TConfig):
        """Returns a copy of `config`"""
        config = copy.deepcopy(config)
        return config

    @staticmethod
    def _resolve_run(run: TConfig | tuple[TConfig, Unpack[TArguments]]):
        if isinstance(run, tuple):
            (config, *args) = run
        else:
            config = cast(TConfig, run)
            args = []
        args = cast(tuple[Unpack[TArguments]], args)
        return (config, args)

    @staticmethod
    def _resolve_runs(runs: list[TConfig] | list[tuple[TConfig, Unpack[TArguments]]]):
        resolved: list[tuple[TConfig, tuple[Unpack[TArguments]]]] = []
        for run in runs:
            resolved.append(Runner._resolve_run(run))

        return resolved

    def local(
        self,
        run: TConfig | tuple[TConfig, Unpack[TArguments]],
        env: dict[str, str] | None = None,
        reset_id: bool = True,
    ):
        config, args = self._resolve_run(run)

        config = self.update_config(config)
        if reset_id:
            config.id = BaseConfig.generate_id(ignore_rng=True)

        env = {**self.DEFAULT_ENV, **(env or {})}
        env_old = {k: os.environ.get(k, None) for k in env}
        os.environ.update(env)
        try:
            self.run(config, *args)
        finally:
            for k, v in env_old.items():
                if v is None:
                    _ = os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    __call__ = local

    def fast_dev_run(
        self,
        runs: list[TConfig] | list[tuple[TConfig, Unpack[TArguments]]],
        env: dict[str, str] | None = None,
        n_batches: int = 1,
        devices: str | int | None = 1,
        stop_on_error: bool = True,
    ):
        """
        Runs a list of configs locally w/ `LightningTrainer.fast_dev_run = True`.
        """

        resolved_runs = self._resolve_runs(runs)
        self._validate_runs(resolved_runs)

        for config, args in tqdm(resolved_runs, desc="Fast dev run"):
            run_id = config.id
            run_name = config.name
            try:
                config = self.update_config(config)
                config.trainer.fast_dev_run = n_batches
                if devices is not None:
                    config.trainer.devices = devices
                self.local((config, *args), env=env, reset_id=True)
            except BaseException as e:
                # print full traceback
                log.critical(f"Error in run with {run_id=} ({run_name=}): {e}")
                traceback.print_exc()
                if stop_on_error:
                    raise

    @staticmethod
    def _validate_runs(runs: list[tuple[TConfig, tuple[Unpack[TArguments]]]]):
        if not runs:
            raise ValueError("No run configs provided.")

        id_counter = Counter(config.id for config, _ in runs if config.id is not None)
        for id, count in id_counter.items():
            if count > 1:
                raise ValueError(f"Duplicate id {id=}")

    @remove_slurm_environment_variables()
    @remove_wandb_environment_variables()
    def submit(
        self,
        runs: list[TConfig] | list[tuple[TConfig, Unpack[TArguments]]],
        *,
        gpus: int,
        nodes: int,
        partition: str,
        cpus_per_task: int,
        snapshot: bool | Path,
        timeout: timedelta = timedelta(hours=72),
        memory: int = 480,
        email: str | None = None,
        volta16gb: bool | None = None,
        volta32gb: bool | None = None,
        slurm_additional_parameters: dict[str, str] | None = None,
        slurm_setup: list[str] | None = None,
        snapshot_base: Path | None = None,
        env: dict[str, str] | None = None,
    ):
        if volta16gb and volta32gb:
            raise ValueError("Cannot have both volta16gb and volta32gb")
        elif volta16gb is None and volta32gb is None:
            volta16gb = False
            volta32gb = True

        if volta16gb is None:
            volta16gb = False
        if volta32gb is None:
            volta32gb = False

        resolved_runs = [
            (self.update_config(c), args) for c, args in self._resolve_runs(runs)
        ]
        self._validate_runs(resolved_runs)

        if snapshot_base is None:
            current_user = getpass.getuser()
            snapshot_base = Path(f"/checkpoint/{current_user}/ll_snapshots/")

        if snapshot is True:
            snapshot = snapshot_modules(
                snapshot_base, ["st", "ll", "submitit"]
            ).absolute()

        env = {**self.DEFAULT_ENV, **(env or {})}

        base_path = Path(".") / "slurm_logs"
        base_path.mkdir(exist_ok=True, parents=True)

        additional_parameters = {}
        if email:
            additional_parameters.update({"mail_user": email, "mail_type": "FAIL"})
        if volta16gb:
            additional_parameters.update({"constraint": "volta16gb"})
        if volta32gb:
            additional_parameters.update({"constraint": "volta32gb"})
        if slurm_additional_parameters:
            additional_parameters.update(slurm_additional_parameters)

        setup = []
        if env:
            setup.extend(f"export {k}={v}" for k, v in env.items())
        if slurm_setup:
            setup.extend(slurm_setup)
        if snapshot:
            snapshot_str = str(snapshot.resolve().absolute())
            setup.append(f"export {self.SNAPSHOT_ENV_NAME}={snapshot_str}")
            setup.append(f"export PYTHONPATH={snapshot_str}:$PYTHONPATH")

        executor = AutoExecutor(folder=base_path / "%j")
        executor.update_parameters(
            name=self.slurm_job_name,
            mem_gb=memory,
            timeout_min=int(timeout.total_seconds() / 60),
            cpus_per_task=cpus_per_task,
            tasks_per_node=gpus,
            gpus_per_node=gpus,
            nodes=nodes,
            slurm_partition=partition,
            slurm_additional_parameters=additional_parameters,
            slurm_setup=setup,
        )

        map_array_args = list(zip(*[(c, *args) for c, args in resolved_runs]))
        log.critical(f"Submitting {len(resolved_runs)} jobs to {partition}.")
        jobs = executor.map_array(self.run, *map_array_args)
        for job, (config, _) in zip(jobs, resolved_runs):
            log.critical(f"[id={config.id}] Submitted job: {job.job_id} to {partition}")
        return jobs
