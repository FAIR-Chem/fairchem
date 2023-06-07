from abc import ABC, abstractmethod
from functools import cache, cached_property
from logging import getLogger
from typing import Any, Generic, Literal, Protocol, final, runtime_checkable

import torch
import torch.func
import torch.nn.functional as F
from einops import reduce
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch, Data
from typing_extensions import TypeVar, override

from ll import BaseConfig as LLBaseConfig
from ll import (
    Field,
    LightningModuleBase,
    Normalizer,
    NormalizerConfig,
    TypedConfig,
)
from ll.data.balanced_batch_sampler import BalancedBatchSampler

from ..datasets.lmdb_dataset import LmdbDataset, LmdbDatasetConfig
from ..datasets.oc22_lmdb_dataset import OC22LmdbDataset, OC22LmdbDatasetConfig
from ..models.gemnet.gemnet import GemNetT, GemNetTConfig
from ..modules.callbacks.ema import EMA, EMAConfig
from ..modules.losses import atomwisel2, l2mae
from ..modules.metrics import S2EFMetrics, S2EFMetricsConfig

log = getLogger(__name__)


class ReduceLROnPlateauConfig(TypedConfig):
    name: Literal["ReduceLROnPlateau"] = "ReduceLROnPlateau"

    monitor: str
    mode: Literal["min", "max"] = "min"
    patience: int = 3
    factor: float = 0.8
    min_lr: float = 0.0
    cooldown: int = 0
    threshold: float = 1.0e-4
    threshold_mode: Literal["rel", "abs"] = "rel"
    eps: float = 1e-8

    interval_frequency: tuple[str, int] | None = None


LRSchedulerConfig = ReduceLROnPlateauConfig


class AdamConfig(TypedConfig):
    name: Literal["Adam"] = "Adam"

    lr: float = 2.0e-4
    amsgrad: bool = False
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


class AdamWConfig(TypedConfig):
    name: Literal["AdamW"] = "AdamW"

    lr: float = 2.0e-4
    amsgrad: bool = False
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


class EnergyLossConfig(TypedConfig):
    coefficient: float = 1.0
    function: Literal["l1", "mae", "mse"] = "mae"


class ForceLossConfig(TypedConfig):
    coefficient: float = 100.0
    function: Literal["l1", "mae", "mse", "l2mae", "atomwisel2"] = "mae"
    free_atoms_only: bool = True


class LossConfig(TypedConfig):
    energy: EnergyLossConfig = EnergyLossConfig()
    force: ForceLossConfig = ForceLossConfig()


class DatasetConfig(TypedConfig):
    dataset: LmdbDatasetConfig | OC22LmdbDatasetConfig = Field(
        discriminator="name"
    )
    shuffle: bool | None = None


class DataConfig(TypedConfig):
    train: DatasetConfig | None = None
    val: DatasetConfig | None = None
    test: DatasetConfig | None = None
    predict: DatasetConfig | None = None

    batch_size: int
    num_workers: int


class NormalizationConfig(TypedConfig):
    energy: NormalizerConfig = NormalizerConfig(
        mean=-0.7554450631141663,
        std=2.887317180633545,
    )
    force: NormalizerConfig = NormalizerConfig(
        std=2.887317180633545,
    )


class BaseConfig(LLBaseConfig):
    data: DataConfig
    normalization: NormalizationConfig = NormalizationConfig()
    optimizer: AdamWConfig | AdamConfig = Field(discriminator="name")
    lr_scheduler: LRSchedulerConfig | None = None
    loss: LossConfig = LossConfig()
    metrics: S2EFMetricsConfig = S2EFMetricsConfig(free_atoms_only=False)

    ema: EMAConfig | None = None


TConfig = TypeVar("TConfig", bound=BaseConfig, infer_variance=True)


class S2EFBaseModule(LightningModuleBase[TConfig], ABC, Generic[TConfig]):
    @override
    def __init__(self, hparams: TConfig):
        super().__init__(hparams)

        if self.config.ema is not None:
            log.critical(f"Using EMA callback with config: {self.config.ema}")
            self.register_callback(EMA(self.config.ema))

        self.train_metrics = S2EFMetrics(self.config.metrics)
        self.val_metrics = S2EFMetrics(self.config.metrics)

    @abstractmethod
    @override
    def forward(self, data: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @cached_property
    def energy_norm(self):
        return Normalizer(self.config.normalization.energy)

    @cached_property
    def force_norm(self):
        return Normalizer(self.config.normalization.force)

    def energy_loss(self, data: Batch, energy: torch.Tensor):
        config = self.config.loss.energy
        # When computing the loss, we compare the model's output
        # to a normalized version of the target.
        energy_target = self.energy_norm.normalize(data.y)

        if self.dev:
            assert (
                energy.shape == energy_target.shape
            ), f"{energy.shape=} != {energy_target.shape=}"

        match config.function:
            case "l1" | "mae":
                energy_loss = F.l1_loss(
                    energy, energy_target, reduction="none"
                )
            case "mse":
                energy_loss = F.mse_loss(
                    energy, energy_target, reduction="none"
                )
            case _:
                raise ValueError(
                    f"Unknown energy loss function: {config.function}"
                )
        # energy_loss: (B)

        # Compute average loss
        energy_loss_numel = energy_loss.numel()
        energy_loss = reduce(energy_loss, "... -> ", "mean")

        # Multiply by energy coeff
        energy_loss = energy_loss * config.coefficient

        self.log("energy_loss", energy_loss, batch_size=energy_loss_numel)
        return energy_loss

    def force_loss(self, data: Batch, forces: torch.Tensor):
        config = self.config.loss.force
        # When computing the loss, we compare the model's output
        # to a normalized version of the target.
        forces_target = self.force_norm.normalize(data.force)

        if self.dev:
            assert (
                forces.shape == forces_target.shape
            ), f"{forces.shape=} != {forces_target.shape=}"

        match config.function:
            case "l1" | "mae":
                force_loss = F.l1_loss(
                    forces, forces_target, reduction="none"
                )  # (N, 3)
            case "mse":
                force_loss = F.mse_loss(
                    forces, forces_target, reduction="none"
                )  # (N, 3)
            case "l2mae":
                force_loss = l2mae(
                    forces, forces_target, reduction="none"
                )  # (N,)
            case "atomwisel2":
                natoms: torch.Tensor = data.natoms  # (B,)
                natoms = natoms[data.batch]  # (N,)
                force_loss = atomwisel2(
                    forces, forces_target, natoms, reduction="none"
                )  # (N,)
            case _:
                raise ValueError(
                    f"Unknown force loss function: {config.function}"
                )

        # Only consider free atoms
        if config.free_atoms_only:
            free_mask: torch.Tensor = ~data.fixed  # (N,)
            force_loss = force_loss[free_mask]

        # Compute average loss
        force_loss_numel = force_loss.numel()
        force_loss = reduce(force_loss, "... -> ", "mean")

        # Multiply by force coeff
        force_loss = force_loss * config.coefficient

        self.log("force_loss", force_loss, batch_size=force_loss_numel)
        return force_loss

    def loss(self, data: Batch, energy: torch.Tensor, forces: torch.Tensor):
        energy_loss = self.energy_loss(data, energy)
        force_loss = self.force_loss(data, forces)
        loss = energy_loss + force_loss

        batch_size = int(data.batch.max() + 1)
        self.log("loss", loss, batch_size=batch_size)
        return loss

    @override
    def training_step(self, batch: Batch, batch_idx: int):
        with self.log_context("train/"):
            energy, forces = self(batch)
            loss = self.loss(batch, energy, forces)

            with torch.no_grad():
                self.log_dict(
                    self.train_metrics(
                        batch,
                        # For metrics, we denormalize the model's output
                        # and compare it to the original target.
                        self.energy_norm.denormalize(energy),
                        self.force_norm.denormalize(forces),
                    )
                )

            return loss

    @override
    def validation_step(self, batch: Batch, batch_idx: int):
        with self.log_context("val/"):
            energy, forces = self(batch)
            self.log_dict(
                self.val_metrics(
                    batch,
                    # For metrics, we denormalize the model's output
                    # and compare it to the original target.
                    self.energy_norm.denormalize(energy),
                    self.force_norm.denormalize(forces),
                )
            )

    @override
    def configure_optimizers(self):
        # Create optimizer based on config
        match self.config.optimizer:
            case AdamConfig() as config:
                optimizer = Adam(
                    self.parameters(),
                    lr=config.lr,
                    amsgrad=config.amsgrad,
                    weight_decay=config.weight_decay,
                )
            case AdamWConfig() as config:
                optimizer = AdamW(
                    self.parameters(),
                    lr=config.lr,
                    amsgrad=config.amsgrad,
                    weight_decay=config.weight_decay,
                )
            case _:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        return_dict: dict[str, Any] = {"optimizer": optimizer}

        # Create learning rate scheduler based on config
        match self.config.lr_scheduler:
            case None:
                pass
            case ReduceLROnPlateauConfig() as config:
                if config.interval_frequency is not None:
                    interval, frequency = config.interval_frequency
                else:
                    # If the interval and frequency of ReduceLROnPlateau
                    # are not explicitly specified, we try to infer them
                    # from the training config. By default, we process
                    # ReduceLROnPlateau every time we do validation.
                    (
                        interval,
                        frequency,
                    ) = (
                        self.determine_reduce_lr_on_plateau_interval_frequency()
                    )

                return_dict["lr_scheduler"] = {
                    "scheduler": ReduceLROnPlateau(
                        optimizer,
                        mode=config.mode,
                        factor=config.factor,
                        patience=config.patience,
                        threshold=config.threshold,
                        threshold_mode=config.threshold_mode,
                        cooldown=config.cooldown,
                        min_lr=config.min_lr,
                        eps=config.eps,
                    ),
                    "monitor": config.monitor,
                    "interval": interval,
                    "frequency": frequency,
                    "strict": True,
                    "reduce_on_plateau": True,
                }
            case _:
                raise ValueError(
                    f"Unknown lr_scheduler: {self.config.lr_scheduler}"
                )

        return return_dict

    def collate_fn(self, data_list: list[Data]):
        # This can be overridden to modify the collation behavior
        batch = Batch.from_data_list(data_list)
        return batch

    def data_transform(self, data: Data):
        # This can be overridden to add per-sample transforms (on the CPU)

        # Set the "fixed" attribute to be of dtype bool (currently, it's of dtype long)
        data.fixed = data.fixed == 1

        return data

    def _dataset(self, config: DatasetConfig):
        # TODO: Teardown the datasets when we're done with them
        match config.dataset:
            case LmdbDatasetConfig():
                dataset = LmdbDataset(config.dataset)
            case OC22LmdbDatasetConfig():
                dataset = OC22LmdbDataset(config.dataset)
            case _:
                raise ValueError(f"Unknown dataset: {config.dataset}")
        dataset = self.dataset(dataset, transform=self.data_transform)
        return dataset

    @cache
    def train_dataset(self):
        if (dataset_config := self.config.data.train) is None:
            raise ValueError("No training dataset specified")
        dataset = self._dataset(dataset_config)
        return dataset

    @override
    def train_dataloader(self):
        if (config := self.config.data.train) is None:
            return None

        dataset = self.train_dataset()
        sampler = self.distributed_sampler(
            dataset,
            shuffle=True if config.shuffle is None else config.shuffle,
        )
        batch_sampler = BalancedBatchSampler(
            sampler,
            batch_size=self.config.data.batch_size,
            device=self.device,
        )
        dataloader = self.dataloader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
        )
        return dataloader

    @cache
    def val_dataset(self):
        if (dataset_config := self.config.data.val) is None:
            raise ValueError("No validation dataset specified")
        dataset = self._dataset(dataset_config)
        return dataset

    @override
    def val_dataloader(self):
        if (config := self.config.data.val) is None:
            return None

        dataset = self.val_dataset()
        sampler = self.distributed_sampler(
            dataset,
            shuffle=False if config.shuffle is None else config.shuffle,
        )
        batch_sampler = BalancedBatchSampler(
            sampler,
            batch_size=self.config.data.batch_size,
            device=self.device,
        )
        dataloader = self.dataloader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
        )
        return dataloader


ModelConfig = GemNetTConfig


class S2EFConfig(BaseConfig):
    model: ModelConfig


@runtime_checkable
class ModelProtocol(Protocol):
    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        ...


@final
class S2EFModule(S2EFBaseModule[S2EFConfig]):
    def construct_model(self):
        # Create the right model based on the config given.
        match self.config.model:
            case GemNetTConfig() as config:
                model = GemNetT(None, 0, 1, **dict(config))
            case _:
                raise ValueError(f"Unknown model: {self.config.model}")

        # Do any additional necessary checks.
        # Make sure that the model returns a tuple of energy and forces.
        if not isinstance(model, ModelProtocol):
            raise ValueError(
                "Model does not implement the ModelProtocol interface. "
                "Make sure that the model's `forward` method returns a tuple of energy and forces."
            )

        return model

    @override
    def __init__(self, hparams: S2EFConfig):
        super().__init__(hparams)

        self.model = self.construct_model()

    @override
    @classmethod
    def config_cls(cls):
        return S2EFConfig

    @override
    def forward(self, data: Batch):
        model_out = self.model(data)
        # Make sure that the model returns a tuple of energy and forces.
        if not isinstance(model_out, tuple):
            raise ValueError(
                "Model does not implement the ModelProtocol interface. "
                "Make sure that the model's `forward` method returns a tuple of energy and forces."
            )
        if len(model_out) != 2:
            raise ValueError(
                f"Expected model to return a tuple of length 2, but got a tuple of length "
                f"{len(model_out)}"
            )

        energy, forces = model_out

        # Any additional post-processing of the model output can be done here.
        # Squeeze the final dimension of the energy tensor.
        energy = energy.squeeze(dim=-1)  # (B, 1) -> (B,)

        return energy, forces
