from dataclasses import dataclass
from typing import Any, Callable, Literal, Union

import torch
from typing_extensions import Annotated, override

from ocpmodels.common.typed_config import Field, TypedConfig
from ocpmodels.models.gemnet_oc_mt.config import (
    BackboneConfig as GOCBackboneConfig,
)

from ...common.utils import MappedKeyType
from .normalizer import NormalizerTargetConfig


# region Output Head Config
class BaseOutputHeadConfig(TypedConfig):
    custom_head: bool = False
    per_task: bool = False

    @override
    def __post_init__(self):
        super().__post_init__()

        if not self.custom_head:
            raise NotImplementedError(
                f"The MT trainer requires all outputs to have custom_head=True."
            )
        if not self.per_task:
            raise NotImplementedError(
                f"The MT trainer requires all outputs to have per_task=True."
            )


class SystemLevelOutputHeadConfig(BaseOutputHeadConfig):
    level: Literal["system"] = "system"


class AtomLevelOutputHeadConfig(BaseOutputHeadConfig):
    level: Literal["atom"] = "atom"
    irrep_dim: int
    train_on_free_atoms: bool = True
    eval_on_free_atoms: bool = True
    use_raw_edge_vecs: bool = False

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.irrep_dim != 1:
            raise NotImplementedError(
                f"Only irrep_dim=1 is supported for the MT trainer."
            )

        if not self.use_raw_edge_vecs:
            raise NotImplementedError(
                f"Only use_raw_edge_vecs=True is supported for the MT trainer."
            )


OutputHeadConfig = Annotated[
    Union[SystemLevelOutputHeadConfig, AtomLevelOutputHeadConfig],
    Field(discriminator="level"),
]

OutputsConfig = Annotated[dict[str, OutputHeadConfig], Field()]
# endregion

# region Loss Config


class LossFnConfig(TypedConfig):
    target: str
    fn: Literal["mae", "mse", "l1", "l2", "l2mae"]

    coefficient: Any | None = None
    reduction: Literal["sum", "mean", "structure_wise_mean"] = "mean"
    per_task: bool = False

    @override
    def __post_init__(self):
        super().__post_init__()

        if not self.per_task:
            raise ValueError(
                f"Loss {self.target} must have per_task=True for MT trainer."
            )

        if self.coefficient is not None:
            raise ValueError(
                f"Loss {self.target} must have coefficient=None for MT trainer. "
                "Use the loss_coefficients in `config.tasks.mt` (MultiTaskConfig) instead."
            )


@dataclass(frozen=True)
class LossFn:
    config: LossFnConfig
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


LossFnsConfig = Annotated[list[LossFnConfig], Field()]

# endregion


# region Dataset Config
SplitDatasetConfig = dict[str, Any]


class TaskDatasetConfig(TypedConfig):
    train: SplitDatasetConfig | None = None
    val: SplitDatasetConfig | None = None
    test: SplitDatasetConfig | None = None

    key_mapping: dict[str, MappedKeyType] = {}

    copy_from_train: bool = True

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.copy_from_train and self.train is not None:
            if self.val is not None:
                self.val = {**self.train, **self.val}
            if self.test is not None:
                self.test = {**self.train, **self.test}

        # Make sure train/val/test don't have "key_mapping"
        # (since it's already been applied)
        for config in [self.train, self.val, self.test]:
            if config and "key_mapping" in config:
                raise ValueError(
                    "Per-split key_mapping is not supported. Please use the task-level key_mapping instead."
                )


class TemperatureSamplingConfig(TypedConfig):
    type: Literal["temperature"] = "temperature"
    temperature: float


class FullyBalancedSamplingConfig(TypedConfig):
    type: Literal["fully_balanced"] = "fully_balanced"


SamplingConfig = Annotated[
    Union[
        TemperatureSamplingConfig,
        FullyBalancedSamplingConfig,
    ],
    Field(discriminator="type"),
]


class OneHotTargetsConfig(TypedConfig):
    graph_level: list[str] = []
    node_level: list[str] = []


class DatasetConfig(TypedConfig):
    datasets: list[TaskDatasetConfig]
    one_hot_targets: OneHotTargetsConfig = OneHotTargetsConfig()
    sampling: SamplingConfig = TemperatureSamplingConfig(temperature=1.0)


# endregion


# region Model Config


class GemNetOCModelConfig(GOCBackboneConfig):
    name: Literal["gemnet_oc_mt"] = "gemnet_oc_mt"


ModelConfig = Annotated[GemNetOCModelConfig, Field(discriminator="name")]

# endregion

# region MultiTask Config


class NormalizerTargetConfig(TypedConfig):
    mean: float = 0.0
    std: float = 1.0


class TaskConfig(TypedConfig):
    idx: int
    name: str
    loss_coefficients: dict[str, float] = {}
    normalization: dict[str, NormalizerTargetConfig] = {}


class MultiTaskConfig(TypedConfig):
    tasks: list[TaskConfig]
    edge_dropout: float | None = None
    node_dropout: float | None = None

    def task_by_name(self, name: str) -> TaskConfig:
        return next(task for task in self.tasks if task.name == name)

    def task_by_idx(self, idx: int) -> TaskConfig:
        return next(task for task in self.tasks if task.idx == idx)


# endregion


def validate_all_configs(
    *,
    dataset: DatasetConfig,
    loss_fns: LossFnsConfig,
    model: ModelConfig,
    outputs: OutputsConfig,
    multi_task: MultiTaskConfig,
):
    num_tasks = len(multi_task.tasks)

    # Validate dataset config
    if len(dataset.datasets) != num_tasks:
        raise ValueError(
            f"Number of datasets ({len(dataset.datasets)}) must match number of tasks ({num_tasks})."
        )

    # Validate loss config
    _ = loss_fns

    # Validate model config
    _ = model

    # Validate outputs config
    for name, output in outputs.items():
        if not output.per_task:
            raise ValueError(f"Output {name} must have per_task=True.")
