import errno
import os
from collections import defaultdict
from functools import cached_property
from logging import getLogger
from typing import Any, Dict, List, cast

import torch
import torch.nn.functional as F
import torchmetrics
import tqdm
from einops import einsum, rearrange, reduce
from torch.nn.parallel.distributed import DistributedDataParallel
from torch_geometric.data import Batch
from torch_scatter import scatter
from typing_extensions import override

from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.data_parallel import OCPDataParallel
from ocpmodels.common.registry import registry
from ocpmodels.common.typed_config import TypeAdapter
from ocpmodels.common.utils import save_checkpoint
from ocpmodels.modules.evaluator import Evaluator

from ..base_trainer import BaseTrainer
from .balanced_batch_sampler import BalancedBatchSampler
from .collate import ParallelCollater
from .config import (
    AtomLevelOutputHeadConfig,
    DatasetConfig,
    LossFn,
    LossFnsConfig,
    ModelConfig,
    MultiTaskConfig,
    OutputsConfig,
    validate_all_configs,
)
from .dataset import create_datasets
from .loss import create_losses
from .normalizer import denormalize_context, denormalize_tensors
from .scaling.compat import load_scales_compat
from .scaling.util import ensure_fitted, load_state_dict

log = getLogger(__name__)


def _safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    b = b.masked_fill(b == 0.0, 1.0)
    return a / b


def _reduce_loss(loss: torch.Tensor, mask: torch.Tensor, reduction: str):
    match reduction:
        case "sum":
            loss = reduce(loss, "b t -> ", "sum")
        case "mean":
            # loss = reduce(loss, "b t -> ", "sum") / reduce(mask, "b t -> ", "sum")
            loss = _safe_divide(
                reduce(loss, "b t -> ", "sum"),
                reduce(mask, "b t -> ", "sum"),
            )
        case "task_mean":
            loss = _safe_divide(
                reduce(loss, "b t -> t", "sum"),
                reduce(mask, "b t -> t", "sum"),
            )
            loss = reduce(loss, "t -> ", "mean")
        case _:
            raise ValueError(f"Unknown redution: {reduction}")

    return loss


@registry.register_trainer("mt")
class MTTrainer(BaseTrainer):
    @override
    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_fns,
        eval_metrics,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
        name="mt",
    ):
        self._train_dataset_sizes: dict[int, int] = {}
        super().__init__(
            task,
            model,
            outputs,
            DatasetConfig.from_dict(
                dataset
            ),  # HACK: wrap it in a class so it doesn't get registered as a dict
            optimizer,
            loss_fns,
            eval_metrics,
            identifier,
            timestamp_id,
            run_dir,
            is_debug,
            print_every,
            seed,
            logger,
            local_rank,
            amp,
            cpu,
            name,
            slurm,
            noddp,
        )

        validate_all_configs(
            dataset=self.dataset_config,
            loss_fns=self.loss_config,
            model=self.model_config,
            outputs=self.typed_output_targets,
            multi_task=self.multi_task_config,
        )

        if self.multi_task_config.lovely_tensors:
            import lovely_tensors

            lovely_tensors.monkey_patch()

    @override
    def train(self, disable_eval_tqdm: bool = False) -> None:
        # Same as the base trainer, except we need to use our own ScaleFactor implementation.

        ensure_fitted(self._unwrapped_model)
        super().train(disable_eval_tqdm)

    def _all_per_task_metric_objects(self):
        keys = [
            "train_per_task_steps",
            "train_per_task_energy_maes",
            "val_per_task_energy_maes",
            "train_per_task_force_maes",
            "val_per_task_force_maes",
        ]

        for key in keys:
            value: dict[int, torchmetrics.Metric] = getattr(self, key)
            metrics: list[torchmetrics.Metric] = []
            for task_idx in range(self.num_tasks):
                metric = value[task_idx]
                metrics.append(metric)
            yield key, metrics

    @override
    def load_checkpoint(self, checkpoint_path: str) -> None:
        # Same as the base trainer, except we need to use our own ScaleFactor implementation.

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        log.info(f"Loading checkpoint from: {checkpoint_path}")
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", None)
        self.primary_metric = checkpoint.get("primary_metric", None)

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module.")
        mod_key_count = next(iter(self.model.state_dict())).count("module.")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        strict = self.config["task"].get("strict_load", True)
        _ = load_state_dict(self.model, new_dict, strict=strict)

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        if "ema" in checkpoint and checkpoint["ema"] is not None:
            assert (
                self.ema is not None
            ), "EMA not initialized but found in checkpoint."
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        scale_dict = checkpoint.get("scale_dict", None)
        if scale_dict:
            log.info(
                "Overwriting scaling factors with those loaded from checkpoint. "
                "If you're generating predictions with a pretrained checkpoint, this is the correct behavior. "
                "To disable this, delete `scale_dict` from the checkpoint. "
            )
            load_scales_compat(self._unwrapped_model, scale_dict)

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(
                    checkpoint["normalizers"][key]
                )
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])

        per_task_metrics = checkpoint.get("per_task_metrics", None)
        if per_task_metrics:
            for key, metrics in self._all_per_task_metric_objects():
                metrics_dict = per_task_metrics.get(key, None)
                if not metrics_dict:
                    continue

                for task_idx, metric in enumerate(metrics):
                    _ = metric.load_state_dict(metrics_dict[task_idx])

    @override
    def save(
        self,
        metrics=None,
        checkpoint_file: str = "checkpoint.pt",
        training_state: bool = True,
    ):
        if not self.is_debug and distutils.is_master():
            if training_state:
                return save_checkpoint(
                    {
                        "epoch": self.epoch,
                        "step": self.step,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.scheduler.state_dict()
                        if self.scheduler.scheduler_type != "Null"
                        else None,
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "val_metrics": metrics,
                        "per_task_metrics": {
                            key: {
                                task_idx: metric.state_dict()
                                for task_idx, metric in enumerate(metrics)
                            }
                            for key, metrics in self._all_per_task_metric_objects()
                        },
                        "ema": self.ema.state_dict() if self.ema else None,
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                        "best_val_metric": self.best_val_metric,
                        "primary_metric": self.evaluation_metrics.get(
                            "primary_metric",
                            self.evaluator.task_primary_metric[self.name],
                        ),
                    },
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
            else:
                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()
                ckpt_path = save_checkpoint(
                    {
                        "state_dict": self.model.state_dict(),
                        "normalizers": {
                            key: value.state_dict()
                            for key, value in self.normalizers.items()
                        },
                        "config": self.config,
                        "val_metrics": metrics,
                        "amp": self.scaler.state_dict()
                        if self.scaler
                        else None,
                    },
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
                if self.ema:
                    self.ema.restore()
                return ckpt_path
        return None

    @cached_property
    def multi_task_config(self):
        return TypeAdapter(MultiTaskConfig).validate_python(
            self.config["task"].get("mt", {})
        )

    @property
    def num_tasks(self):
        return len(self.multi_task_config.tasks)

    @override
    def load_task(self):
        # We don't use this way of normalizing.
        self.normalizers = {}

        self.typed_output_targets = outputs_config = TypeAdapter(
            OutputsConfig
        ).validate_python(self.config["outputs"])
        self.multi_task_targets = defaultdict[str, list[str]](lambda: [])
        self.per_task_output_targets = {}
        self.output_targets = {}
        for target_name, target_config in outputs_config.items():
            target_config_dict = target_config.to_dict()
            self.output_targets[target_name] = target_config_dict.copy()
            for task_idx in range(self.num_tasks):
                key = f"{target_name}_task_{task_idx}"
                self.per_task_output_targets[key] = target_config_dict.copy()
                self.multi_task_targets[target_name].append(key)

        self.evaluation_metrics = self.config.get("eval_metrics", {})
        self.evaluator = Evaluator(
            task=self.name,
            eval_metrics=self.evaluation_metrics.get(
                "metrics", Evaluator.task_metrics.get(self.name, {})
            ),
        )

        if self.multi_task_config.log_task_steps_and_epochs:
            self.train_per_task_steps: dict[int, torchmetrics.SumMetric] = {}
            for task_idx in range(self.num_tasks):
                metric = torchmetrics.SumMetric().to(self.device)
                metric.persistent(True)
                self.train_per_task_steps[task_idx] = metric
        else:
            self.train_per_task_steps = None

        self.train_per_task_energy_maes = {
            task_idx: torchmetrics.MeanAbsoluteError().to(self.device)
            for task_idx in range(self.num_tasks)
        }
        self.val_per_task_energy_maes = {
            task_idx: torchmetrics.MeanAbsoluteError().to(self.device)
            for task_idx in range(self.num_tasks)
        }
        self.train_per_task_force_maes = {
            task_idx: torchmetrics.MeanAbsoluteError().to(self.device)
            for task_idx in range(self.num_tasks)
        }
        self.val_per_task_force_maes = {
            task_idx: torchmetrics.MeanAbsoluteError().to(self.device)
            for task_idx in range(self.num_tasks)
        }

    @cached_property
    def model_config(self):
        model_config_dict: dict = self.config["model_attributes"].copy()
        model_config_dict["name"] = self.config["model"]
        model_config_dict = self.multi_task_config.update_model_config_dict(
            model_config_dict
        )

        return TypeAdapter(ModelConfig).validate_python(model_config_dict)

    @override
    def load_model(self) -> None:
        # Build model
        if distutils.is_master():
            log.info(f"Loading model: {self.config['model']}")

        self.model = registry.get_model_class(self.config["model"])(
            self.per_task_output_targets,
            self.model_config,
            **self.config["model_attributes"],
        ).to(self.device)

        if distutils.is_master():
            log.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )

    @property
    def dataset_config(self):
        dataset_config = self.config["dataset"]
        assert isinstance(
            dataset_config, DatasetConfig
        ), f"{dataset_config=} is not a DatasetConfig"
        return dataset_config

    @override
    def get_sampler(
        self, dataset, batch_size: int, shuffle: bool
    ) -> BalancedBatchSampler:
        balancing_mode = self.config["optim"].get("load_balancing", None)
        on_error = self.config["optim"].get("load_balancing_on_error", None)
        if balancing_mode is not None:
            if on_error is None:
                on_error = "raise"
        else:
            balancing_mode = "atoms"

        if on_error is None:
            on_error = "warn_and_no_balance"

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
            raise NotImplementedError("GP not implemented for MT/FT.")
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()

        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            on_error=on_error,
        )
        return sampler

    @override
    def load_datasets(self) -> None:
        log.info("Loading datasets")
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            self.dataset_config,
            self.config["model_attributes"].get("otf_graph", False),
        )

        batch_size = self.config["optim"]["batch_size"]
        assert isinstance(batch_size, int), f"{batch_size=} is not an integer"
        eval_batch_size = self.config["optim"].get(
            "eval_batch_size", batch_size
        )
        assert isinstance(
            eval_batch_size, int
        ), f"{eval_batch_size=} is not an integer"

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ), train_dataset_sizes = create_datasets(
            self.dataset_config,
            self.multi_task_config,
            self.model_config,
        )
        if train_dataset_sizes:
            for task, size in zip(
                self.multi_task_config.tasks, train_dataset_sizes
            ):
                self._train_dataset_sizes[task.idx] = size

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if self.train_dataset is not None:
            self.train_sampler = self.get_sampler(
                self.train_dataset, batch_size, shuffle=True
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset, self.train_sampler
            )

        if self.val_dataset is not None:
            self.val_sampler = self.get_sampler(
                self.val_dataset, eval_batch_size, shuffle=False
            )
            self.val_loader = self.get_dataloader(
                self.val_dataset, self.val_sampler
            )

        if self.test_dataset is not None:
            self.test_sampler = self.get_sampler(
                self.test_dataset, eval_batch_size, shuffle=False
            )
            self.test_loader = self.get_dataloader(
                self.test_dataset, self.test_sampler
            )
        # load relaxation dataset
        if "relax_dataset" in self.config["task"]:
            raise NotImplementedError(
                "Relaxation dataset not implemented for MT."
            )

    @cached_property
    def loss_config(self):
        return TypeAdapter(LossFnsConfig).validate_python(
            self.config["loss_fns"]
        )

    @override
    def load_loss(self) -> None:
        self.loss_fns = create_losses(self.loss_config)

    def _apply_loss_coefficient(
        self,
        loss: torch.Tensor,
        loss_info: LossFn,
    ):
        # loss: (bsz, t) or (n_atoms, t)
        coeff = loss.new_tensor(
            [
                self.multi_task_config.task_by_idx(
                    task_idx
                ).loss_coefficients.get(loss_info.config.target, 1.0)
                for task_idx in range(self.num_tasks)
            ]
        )  # (t,)

        loss = loss * coeff  # (bsz, t) or (n_atoms, t)
        return loss

    @override
    def _compute_loss(
        self,
        out: Dict[str, torch.Tensor],
        batch_list: list[Batch],
    ):
        batch_idx = torch.cat(
            [batch.batch.to(self.device) for batch in batch_list]
        )  # (n_atoms,)
        fixed = torch.cat(
            [batch.fixed.to(self.device) for batch in batch_list]
        )
        free_mask = fixed == 0  # (n_atoms,)

        task_mask = torch.cat(
            [batch.task_mask.to(self.device) for batch in batch_list]
        )  # (bsz, t)

        one_hot = F.one_hot(
            torch.cat(
                [batch.task_idx.to(self.device) for batch in batch_list]
            ),
            num_classes=self.num_tasks,
        ).float()  # (bsz, t)
        one_hot_node = one_hot[batch_idx]  # (n_atoms, t)

        losses: list[torch.Tensor] = []
        for loss_info in self.loss_fns:
            target_name = loss_info.config.target
            output_target = self.typed_output_targets[target_name]
            level = output_target.level

            target = torch.cat(
                [batch[target_name].to(self.device) for batch in batch_list],
                dim=0,
            )  # (bsz,) or (n_atoms, 3)
            # Turn to one-hot
            if level == "system":
                target = einsum(target, one_hot, "b, b t -> b t")
            elif level == "atom":
                target = einsum(target, one_hot_node, "n p, n t -> n t p")
            else:
                raise ValueError(f"Unknown level: {level}")

            pred = out[target_name]  # (bsz, t) or (n_atoms, t, 3)
            mask = task_mask  # (bsz, t)
            if level == "atom":
                mask = mask[batch_idx]  # (n_atoms, t)

            if (
                level == "atom"
                and isinstance(output_target, AtomLevelOutputHeadConfig)
                and output_target.train_on_free_atoms
            ):
                mask = mask & rearrange(free_mask, "n -> n 1")  # (n_atoms, t)

            normalizer = self.normalizers.get(target_name, False)
            if normalizer:
                target = normalizer.norm(target)

            # Compute the loss
            loss = loss_info.fn(pred, target)  # (bsz, t) or (n_atoms, t)

            # Apply mask
            loss = loss * mask

            # Apply the coefficient
            loss = self._apply_loss_coefficient(loss, loss_info)

            # For SWL, we want to reduce the loss per structure, not per atom.
            reduction = loss_info.config.reduction
            if reduction == "structure_wise_mean":
                assert level == "atom", f"{level=} must be atom"
                loss = scatter(loss, batch_idx, dim=0, reduce="sum")  # (b, t)
                force_loss_mask_natoms = scatter(
                    mask.float(), batch_idx, dim=0, reduce="sum"
                )  # (b, t)
                loss = _safe_divide(loss, force_loss_mask_natoms)  # (b, t)
                mask = force_loss_mask_natoms > 0.0  # (b, t)

                # We can now just reduce the loss as normal
                reduction = "mean"

            # Now, reduce the loss
            loss = _reduce_loss(loss, mask, reduction)
            losses.append(loss)

        # Sanity check to make sure the compute graph is correct.
        for lc in losses:
            assert hasattr(lc, "grad_fn")

        loss = sum(losses)
        loss = cast(torch.Tensor, loss)
        return loss

    def _per_task_step_epoch_metrics(self, batch_list: List[Batch]):
        metrics: dict[str, Any] = {}
        if (
            not self.multi_task_config.log_task_steps_and_epochs
            or not self._train_dataset_sizes
        ):
            return metrics

        task_mask = torch.cat(
            [batch.task_mask.to(self.device) for batch in batch_list]
        )  # (b, t)
        task_idx = reduce(task_mask, "b t -> t", "sum")  # (t,)
        for task in self.multi_task_config.tasks:
            metric = self.train_per_task_steps[task.idx]
            metric(task_idx[task.idx])

            step = metric.compute().item()
            metrics = self.evaluator.update(f"{task.name}_step", step, metrics)

            epoch = step / self._train_dataset_sizes[task.idx]
            metrics = self.evaluator.update(
                f"{task.name}_epoch", epoch, metrics
            )

        return metrics

    def _compute_per_task_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch_list: List[Batch],
        validate: bool = False,
    ):
        # HACK: If the model is in eval mode, then we're validating and should keep metric states.
        # Otherwise, we just report the metric values for the current batch.
        per_task_energy_maes = (
            self.val_per_task_energy_maes
            if validate
            else self.train_per_task_energy_maes
        )
        per_task_force_maes = (
            self.val_per_task_force_maes
            if validate
            else self.train_per_task_force_maes
        )

        metrics: Dict[str, Any] = {}
        if not validate:
            metrics.update(self._per_task_step_epoch_metrics(batch_list))
        one_hot = F.one_hot(
            torch.cat(
                [batch.task_idx.to(self.device) for batch in batch_list]
            ),
            num_classes=self.num_tasks,
        ).float()  # (bsz, t)
        energy_target = torch.cat(
            [batch.energy.to(self.device) for batch in batch_list]
        )  # (bsz,)
        energy_target = einsum(
            energy_target, one_hot, "b, b t -> b t"
        )  # (bsz, t)

        batch_idx = torch.cat(
            [batch.batch.to(self.device) for batch in batch_list]
        )
        one_hot_node = one_hot[batch_idx]  # (n_atoms, t)
        forces_target = torch.cat(
            [batch.forces.to(self.device) for batch in batch_list]
        )  # (n_atoms, 3)
        forces_target = einsum(
            forces_target, one_hot_node, "n p, n t -> n t p"
        )  # (n_atoms, t, 3)

        for task in self.multi_task_config.tasks:
            mask = torch.cat(
                [batch.task_mask.to(self.device) for batch in batch_list]
            ) & rearrange(
                torch.cat(
                    [batch.task_idx.to(self.device) for batch in batch_list]
                )
                == task.idx,
                "b -> b 1",
            )  # (bsz, t)
            if not mask.any():
                continue

            node_mask = mask[batch_idx]  # (n_atoms, t)

            energy_mae = per_task_energy_maes[task.idx](
                outputs["energy"][mask], energy_target[mask]
            )
            if not validate:
                metrics = self.evaluator.update(
                    f"{task.name}_energy_mae", energy_mae.item(), metrics
                )

            forces_mae = per_task_force_maes[task.idx](
                outputs["forces"][node_mask], forces_target[node_mask]
            )
            if not validate:
                metrics = self.evaluator.update(
                    f"{task.name}_forces_mae", forces_mae.item(), metrics
                )

        return metrics

    @override
    @torch.no_grad()
    @torch.inference_mode()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)

        # validate as normal
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master():
            log.info(f"Evaluating on {split}.")

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        metrics = {}
        evaluator = Evaluator(
            task=self.name,
            eval_metrics=self.evaluation_metrics.get(
                "metrics", Evaluator.task_metrics.get(self.name, {})
            ),
        )

        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(
                out,
                batch,
                evaluator,
                metrics,
                True,
            )
            metrics = evaluator.update("loss", loss.item(), metrics)

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        # For the per-task metrics, we compute and return the state here
        # so that we can report the final metric.
        for task in self.multi_task_config.tasks:
            energy_mae = self.val_per_task_energy_maes[task.idx]
            metrics = self.evaluator.update(
                f"{task.name}_energy_mae",
                energy_mae.compute().item(),
                metrics,
            )
            energy_mae.reset()

            force_mae = self.val_per_task_force_maes[task.idx]
            metrics = self.evaluator.update(
                f"{task.name}_forces_mae",
                force_mae.compute().item(),
                metrics,
            )
            force_mae.reset()

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            log.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        return metrics

    @override
    def _compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch_list: List[Batch],
        evaluator: Evaluator,
        metrics: Dict[str, Any] = {},
        validate: bool = False,
    ):
        # First, compute the aggregate metrics across all tasks
        aggregated_outputs: Dict[str, torch.Tensor] = {}
        batch_idx = torch.cat(
            [batch.batch.to(self.device) for batch in batch_list]
        )  # (n_atoms,)
        task_mask = torch.cat(
            [batch.task_mask.to(self.device) for batch in batch_list]
        )  # (bsz, t)
        task_mask_node = task_mask[batch_idx]  # (n_atoms, t)
        for target_name, target_value in outputs.items():
            output_target = self.typed_output_targets[target_name]
            level = output_target.level

            if level == "system":
                # Shape should be (bsz, n_tasks)
                aggregated_outputs[target_name] = target_value[
                    task_mask
                ]  # (bsz)
            elif level == "atom":
                # Shape should be (n_atoms, n_tasks, 3)
                aggregated_outputs[target_name] = target_value[
                    task_mask_node
                ]  # (n_atoms, 3)
            else:
                raise NotImplementedError(f"{level=} not implemented.")

        with denormalize_context(
            batch_list,
            [aggregated_outputs],
            [outputs],
        ) as (
            denormed_batch_list,
            [denormed_aggregated_outputs],
            [denormed_outputs],
        ):
            metrics = super()._compute_metrics(
                denormed_aggregated_outputs.copy(),
                denormed_batch_list,
                evaluator,
                metrics,
            )

            # Now, compute the per-task metrics
            _ = metrics.update(
                self._compute_per_task_metrics(
                    denormed_outputs,
                    denormed_batch_list,
                    validate,
                )
            )

        if (
            self.model.training
            and self.step % self.config["cmd"]["print_every"] == 0
        ):
            # Reset the train per-task metrics
            for task_idx in range(self.num_tasks):
                self.train_per_task_energy_maes[task_idx].reset()
                self.train_per_task_force_maes[task_idx].reset()

        return metrics

    def _merge_mt_outputs(self, outputs: Dict[str, torch.Tensor]):
        merged_outputs: Dict[str, torch.Tensor] = outputs.copy()

        # Merge the per-task outputs
        for target_name, target_key_list in self.multi_task_targets.items():
            merged_outputs[target_name] = torch.stack(
                [merged_outputs.pop(key) for key in target_key_list], dim=1
            )

        return merged_outputs

    def _validate_mt_outputs(self, outputs: Dict[str, torch.Tensor]):
        for target, config in self.output_targets.items():
            # Get the value
            value = outputs.get(target, None)
            if value is None:
                log.warning(f"{target=} not in {outputs=}")
                continue

            system = config.get("level", "system")
            if system == "system":
                # Shape should be (bsz, n_tasks)
                assert (
                    value.ndim == 2 and value.shape[1] == self.num_tasks
                ), f"System output {value.shape=} should be (bsz, {self.num_tasks=})"
            elif system == "atom":
                # Shape should be (n_atoms, n_tasks, 3)
                assert (
                    value.ndim == 3 and value.shape[1] == self.num_tasks
                ), f"Atom output {value.shape=} should be (n_atoms, {self.num_tasks=}, 3)"
            else:
                raise NotImplementedError(f"{system=} not implemented.")

    @override
    def _forward(self, batch_list):
        outputs = super()._forward(batch_list)
        outputs = self._merge_mt_outputs(outputs)
        self._validate_mt_outputs(outputs)
        return outputs

    @override
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file: str | None = None,
        disable_tqdm: bool = False,
    ):
        raise NotImplementedError(
            "Predict not implemented for MT pretraining."
        )
