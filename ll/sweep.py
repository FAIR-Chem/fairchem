import copy
import json
import os
from contextlib import contextmanager
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Generic, Literal, TypedDict, Union

import wandb
from typing_extensions import NotRequired, Unpack, override

from .runner import Runner, RunProtocol, TArguments, TConfig
from .util.environment import (
    remove_slurm_environment_variables,
    remove_wandb_environment_variables,
)
from .util.slurm import create_executor

log = getLogger(__name__)

ParameterValue = float | int


# https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#parameters
class Parameter(TypedDict):
    values: NotRequired[list[Any]]
    """Specifies all valid values for this hyperparameter. Compatible with `grid`."""

    value: NotRequired[Any]
    """Specifies the single valid value for this hyperparameter. Compatible with `grid`."""

    distribution: NotRequired[
        Literal[
            "constant",
            "categorical",
            "int_uniform",
            "uniform",
            "q_uniform",
            "log_uniform",
            "log_uniform_values",
            "q_log_uniform",
            "q_log_uniform_values",
            "inv_log_uniform",
            "inv_log_uniform_values",
            "normal",
            "q_normal",
            "log_normal",
            "q_log_normal",
        ]
    ]
    """
    Selects a distribution from the distribution table below. If not specified, will default to `categorical` if `values` is set, to `int_uniform` if `max` and `min` are set to integers, to `uniform` if `max` and `min` are set to floats, or to `constant` if `value` is set.

    https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#distribution

    Valid values for `distribution` are:
        `constant` - Constant distribution. Must specify `value`.

        `categorical` - Categorical distribution. Must specify `values`.

        `int_uniform` - Discrete uniform distribution on integers. Must specify `max` and `min` as integers.

        `uniform` - Continuous uniform distribution. Must specify `max` and `min` as floats.

        `q_uniform` - Quantized uniform distribution. Returns `round(X / q) * q` where X is uniform. `q` defaults to `1`.

        `log_uniform` - Log-uniform distribution. Returns a value `X` between `exp(min)` and `exp(max)` such that the natural logarithm is uniformly distributed between `min` and `max`.

        `log_uniform_values` - Log-uniform distribution. Returns a value `X` between `min` and `max` such that `log(X)` is uniformly distributed between `log(min)` and `log(max)`.

        `q_log_uniform` - Quantized log uniform. Returns `round(X / q) * q` where X is log_uniform. `q` defaults to `1`.

        `q_log_uniform_values` - Quantized log uniform. Returns `round(X / q) * q` where `X` is `log_uniform_values`. `q` defaults to `1`.

        `inv_log_uniform` - Inverse log uniform distribution. Returns `X`, where `log(1/X)` is uniformly distributed between `min` and `max`.

        `inv_log_uniform_values` - Inverse log uniform distribution. Returns `X`, where `log(1/X)` is uniformly distributed between log(1/max) and log(1/min).

        `normal` - Normal distribution. Return value is normally-distributed with mean mu (default 0) and standard deviation sigma (default 1).

        `q_normal` - Quantized normal distribution. Returns `round(X / q) * q` where `X` is `normal`. `q` defaults to `1`.

        `log_normal` - Log normal distribution. Returns a value X such that the natural logarithm `log(X)` is normally distributed with mean `mu` (default `0`) and standard deviation `sigma` (default `1`).

        `q_log_normal` - Quantized log normal distribution. Returns `round(X / q) * q` where `X` is `log_normal`. `q` defaults to `1`.
    """

    probabilities: NotRequired[list[float]]
    """Specify the probability of selecting each element of values when using random."""

    min: NotRequired[ParameterValue]
    """Minimum values. If int, for `int_uniform`-distributed hyperparameters. If float, for `uniform`-distributed hyperparameters."""

    max: NotRequired[ParameterValue]
    """Maximum values. If int, for `int_uniform`-distributed hyperparameters. If float, for `uniform`-distributed hyperparameters."""

    mu: NotRequired[ParameterValue]
    """Mean parameter for `normal`-or-`lognormal`-distributed hyperparameters."""

    sigma: NotRequired[ParameterValue]
    """Standard deviation parameter for `normal`-or-`lognormal`-distributed hyperparameters."""

    q: NotRequired[ParameterValue]
    """Quantization step size for quantized hyperparameters."""

    parameters: NotRequired[dict[str, "Parameters"]]
    """Nest other parameters inside a root level parameter."""


Parameters = dict[str, Union["Parameter", "Parameters"]]


# https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#metric
class Metric(TypedDict):
    name: str
    """Name of the metric to optimize."""

    goal: NotRequired[Literal["minimize", "maximize"]]
    """Whether to minimize or maximize the metric (Default is `minimize`)."""

    target: NotRequired[float]
    """
    Goal value for the metric you're optimizing.
    When any run in the sweep achieves that target value, the sweep's state will be set to `finished`.
    This means all agents with active runs will finish those jobs, but no new runs will be launched in the sweep.
    """


# https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#early_terminate
class EarlyTerminate(TypedDict):
    type: Literal["hyperband"]
    """Specify the stopping algorithm to use. Currently only `hyperband` is supported."""

    min_iter: NotRequired[int]
    """Specify the stopping algorithm"""

    max_iter: NotRequired[int]
    """Specify the maximum number of iterations."""

    s: NotRequired[int]
    """Specify the total number of brackets (required for `max_iter`)."""

    eta: NotRequired[int]
    """Specify the bracket multiplier schedule (default: `3`)."""


# https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
class Configuration(TypedDict):
    method: Literal["grid", "random", "bayes"]
    """
    https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#method

    grid - Iterate over every combination of hyperparameter values. Can be computationally costly.

    random - Choose a random set of hyperparameter values on each iteration based on provided distributions.

    bayes - Create a probabilistic model of a metric score as a function of the hyperparameters, and choose parameters with high probability of improving the metric. Bayesian hyperparameter search method uses a Gaussian Process to model the relationship between the parameters and the model metric and chooses parameters to optimize the probability of improvement. This strategy requires the metrickey to be specified. Works well for small numbers of continuous parameters but scales poorly.
    """

    parameters: Parameters
    """
    https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#parameters

    Describe the hyperparameters to explore during the sweep. For each hyperparameter, specify the name and the possible values as a list of constants (for any `method`) or specify a `distribution` for `random` or `bayes`.
    """

    metric: NotRequired[Metric]
    """
    https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#metric

    Describes the metric to optimize. This metric should be logged explicitly to W&B by your training script.
    """

    early_terminate: NotRequired[EarlyTerminate]
    """
    https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#early_terminate

    Early termination is an optional feature that speeds up hyperparameter search by stopping poorly-performing runs.
    When the early stopping is triggered, the agent stops the current run and gets the next set of hyperparameters to try.
    """

    name: NotRequired[str]
    """The name of the sweep, displayed in the W&B UI."""

    description: NotRequired[str]
    """A description of the sweep, displayed in the W&B UI."""

    project: NotRequired[str]
    """Specify the project for this sweep."""

    entity: NotRequired[str]
    """Specify the entity for this sweep."""

    run_cap: NotRequired[int]
    """Specify the maximum number of runs to create in this sweep."""


class ParameterConstructor:
    def __call__(self, **kwargs: Unpack[Parameter]) -> Parameter:
        return Parameter(**kwargs)

    def constant(self, *, value: Any) -> Parameter:
        return self(distribution="constant", value=value)

    def categorical(
        self, *, values: list[Any], probabilities: list[float] | None = None
    ) -> Parameter:
        if probabilities is not None:
            return self(
                values=values,
                probabilities=probabilities,
            )
        else:
            return self(distribution="categorical", values=values)

    def int_uniform(self, *, min: int, max: int) -> Parameter:
        return self(distribution="int_uniform", min=min, max=max)

    def uniform(self, *, min: float, max: float) -> Parameter:
        return self(distribution="uniform", min=min, max=max)

    def q_uniform(self, *, min: float, max: float, q: float = 1.0) -> Parameter:
        return self(distribution="q_uniform", min=min, max=max, q=q)

    def log_uniform(self, *, min: float, max: float) -> Parameter:
        return self(distribution="log_uniform", min=min, max=max)

    def log_uniform_values(self, *, min: float, max: float) -> Parameter:
        return self(distribution="log_uniform_values", min=min, max=max)

    def q_log_uniform(self, *, min: float, max: float, q: float = 1.0) -> Parameter:
        return self(distribution="q_log_uniform", min=min, max=max, q=q)

    def q_log_uniform_values(
        self, *, min: float, max: float, q: float = 1.0
    ) -> Parameter:
        return self(distribution="q_log_uniform_values", min=min, max=max, q=q)

    def inv_log_uniform(self, *, min: float, max: float) -> Parameter:
        return self(distribution="inv_log_uniform", min=min, max=max)

    def inv_log_uniform_values(self, *, min: float, max: float) -> Parameter:
        return self(distribution="inv_log_uniform_values", min=min, max=max)

    def normal(self, *, mu: float = 0.0, sigma: float = 1.0) -> Parameter:
        return self(distribution="normal", mu=mu, sigma=sigma)

    def q_normal(
        self, *, mu: float = 0.0, sigma: float = 1.0, q: float = 1.0
    ) -> Parameter:
        return self(distribution="q_normal", mu=mu, sigma=sigma, q=q)

    def log_normal(self, *, mu: float = 0.0, sigma: float = 1.0) -> Parameter:
        return self(distribution="log_normal", mu=mu, sigma=sigma)

    def q_log_normal(
        self, *, mu: float = 0.0, sigma: float = 1.0, q: float = 1.0
    ) -> Parameter:
        return self(distribution="q_log_normal", mu=mu, sigma=sigma, q=q)


class Sweep(Generic[TConfig, Unpack[TArguments]]):
    parameter = ParameterConstructor()

    @staticmethod
    def metric(**kwargs: Unpack[Metric]) -> Metric:
        return Metric(**kwargs)

    @staticmethod
    def early_terminate(**kwargs: Unpack[EarlyTerminate]) -> EarlyTerminate:
        return EarlyTerminate(**kwargs)

    @staticmethod
    def config(**kwargs: Unpack[Configuration]) -> Configuration:
        return Configuration(**kwargs)

    @override
    def __init__(
        self,
        base_config: TConfig | tuple[TConfig, Unpack[TArguments]],
        *,
        entity: str | None = None,
        project: str | None = None,
        auto_wrap_run: bool | None = None,
    ):
        self.base_config, self.base_config_args = Runner._resolve_run(base_config)
        self.entity = entity
        self.project = project or self.base_config.project
        self.auto_wrap_run = auto_wrap_run

    def _validate_config(self, config: Configuration):
        if config["method"] == "bayes" and config.get("metric") is None:
            raise ValueError(
                "metric must be specified when using bayes method for sweep"
            )
        if config.get("early_terminate") and config.get("metric") is None:
            raise ValueError(
                "metric must be specified when using early_terminate for sweep"
            )

    def create(self, config: Configuration | Callable[[], Configuration]):
        if callable(config):
            config = config()

        self._validate_config(config)
        return wandb.sweep(config, entity=self.entity, project=self.project)

    @staticmethod
    def _update(config: TConfig, key: str, value: Any):
        # key is a dot-separated string, e.g. "optimizer.lr"
        keys = key.split(".")
        if len(keys) == 1:
            setattr(config, key, value)
        else:
            subconfig = getattr(config, keys[0])
            for k in keys[1:-1]:
                subconfig = getattr(subconfig, k)
            setattr(subconfig, keys[-1], value)

    def _create_sweep_config(
        self,
        sweep_config: dict[str, Any],
        id: str,
    ):
        config = copy.deepcopy(self.base_config)
        config.id = id
        for k, v in sweep_config.items():
            self._update(config, k, v)

        args = copy.deepcopy(self.base_config_args)
        return config, args

    @staticmethod
    @contextmanager
    def _set_sweep_env(id: str, config: Any):
        og_env = os.environ.copy()
        os.environ["LL_WANDB_SWEEP_ID"] = id
        os.environ["LL_WANDB_SWEEP_CONFIG"] = json.dumps(config)
        try:
            yield
        finally:
            for k in ["LL_WANDB_SWEEP_ID", "LL_WANDB_CONFIG_PATHS"]:
                if k not in og_env:
                    _ = os.environ.pop(k, None)
                else:
                    os.environ[k] = og_env[k]

    def local(
        self,
        run: RunProtocol[TConfig, Unpack[TArguments]],
        sweep_id: str,
        *,
        env: dict[str, str] | None = None,
    ):
        def main():
            log.critical(f"Starting agent for sweep {sweep_id}...")

            experiment = wandb.init(
                entity=self.entity,
                project=self.project,
            )
            if experiment is None:
                raise RuntimeError("Failed to initialize wandb")

            sweep_config = wandb.config
            run_id = experiment.id
            assert isinstance(run_id, str), "run_id must be a string"

            with self._set_sweep_env(sweep_id, dict(experiment.config_static.__dict__)):
                log.critical(
                    f"[Sweep {sweep_id}]: Received new run ({run_id=}) for sweep {sweep_id}. Config: {sweep_config}"
                )
                config, args = self._create_sweep_config(sweep_config, id=run_id)

                runner = Runner(run, auto_wrap_run=self.auto_wrap_run)
                return runner((config, *args), env=env, reset_id=False)

        return wandb.agent(
            sweep_id,
            function=main,
            entity=self.entity,
            project=self.project,
        )

    @remove_slurm_environment_variables()
    @remove_wandb_environment_variables()
    def submit(
        self,
        run: RunProtocol[TConfig, Unpack[TArguments]],
        num_agents: int,
        sweep_id: str,
        *,
        nodes: int = 1,
        tasks_per_node: int,
        cpus_per_task: int,
        gpus_per_task: int,
        partition: str,
        timeout: timedelta = timedelta(hours=72),
        memory: int = 480,
        email: str | None = None,
        constraints: list[str] | None = None,
        volta16gb: bool | None = None,
        volta32gb: bool | None = None,
        slurm_additional_parameters: dict[str, str] | None = None,
        slurm_setup: list[str] | None = None,
        snapshot: bool | Path,
        snapshot_base: Path | None = None,
        env: dict[str, str] | None = None,
        job_name_prefix: str = "llsweep_",
        snapshot_env_name: str = "LL_SNAPSHOT",
    ):
        if nodes > 1:
            raise NotImplementedError("Multi-node sweeps are not yet supported")

        executor = create_executor(
            tasks_per_node=tasks_per_node,
            gpus_per_task=gpus_per_task,
            cpus_per_task=cpus_per_task,
            nodes=nodes,
            partition=partition,
            timeout=timeout,
            memory=memory,
            email=email,
            constraints=constraints,
            volta16gb=volta16gb,
            volta32gb=volta32gb,
            slurm_additional_parameters=slurm_additional_parameters,
            slurm_setup=slurm_setup,
            snapshot=snapshot,
            snapshot_base=snapshot_base,
            env=env,
            job_name=f"{job_name_prefix}{sweep_id}",
            snapshot_env_name=snapshot_env_name,
        )

        log.critical(f"Submitting {num_agents} agents to {partition}.")
        jobs = executor.map_array(
            self.local,
            [run] * num_agents,
            [sweep_id] * num_agents,
        )
        for job in jobs:
            log.info(f"[{sweep_id=}] Submitted job: {job.job_id} to {partition}")
        return jobs
