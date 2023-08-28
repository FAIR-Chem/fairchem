from collections import defaultdict
from functools import cached_property
from logging import getLogger
from typing import Any, Dict, List, cast

import torch
import torchmetrics
from einops import rearrange, reduce
from torch.nn.parallel.distributed import DistributedDataParallel
from torch_geometric.data import Batch
from torch_scatter import scatter
from typing_extensions import override

from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollater
from ocpmodels.common.registry import registry
from ocpmodels.common.typed_config import TypeAdapter
from ocpmodels.modules.evaluator import Evaluator

from ..base_trainer import BaseTrainer
from .balanced_batch_sampler import BalancedBatchSampler
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
        if model.get("regress_forces", True) or model.get(
            "direct_forces", False
        ):
            raise NotImplementedError(
                "regress_forces and direct_forces are not supported for MTTrainer"
            )

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
        ) = create_datasets(self.dataset_config, self.multi_task_config)

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

        losses: list[torch.Tensor] = []
        for loss_info in self.loss_fns:
            target_name = loss_info.config.target
            output_target = self.typed_output_targets[target_name]
            level = output_target.level

            target = torch.cat(
                [
                    batch[f"{target_name}_onehot"].to(self.device)
                    for batch in batch_list
                ],
                dim=0,
            )  # (bsz, t) or (n_atoms, t, 3)
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

    def _compute_per_task_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch_list: List[Batch],
    ):
        # HACK: If the model is in eval mode, then we're validating and should keep metric states.
        # Otherwise, we just report the metric values for the current batch.
        is_eval = not self.model.training
        per_task_energy_maes = (
            self.val_per_task_energy_maes
            if is_eval
            else self.train_per_task_energy_maes
        )
        per_task_force_maes = (
            self.val_per_task_force_maes
            if is_eval
            else self.train_per_task_force_maes
        )

        metrics: Dict[str, Any] = {}
        for task_idx in range(self.num_tasks):
            mask = torch.cat(
                [batch.task_mask.to(self.device) for batch in batch_list]
            ) & rearrange(
                torch.cat(
                    [batch.task_idx.to(self.device) for batch in batch_list]
                )
                == task_idx,
                "b -> b 1",
            )  # (bsz, t)
            if not mask.any():
                continue

            node_mask = mask[
                torch.cat(
                    [batch.batch.to(self.device) for batch in batch_list]
                )
            ]  # (n_atoms, t)

            energy_target = torch.cat(
                [batch.energy_onehot.to(self.device) for batch in batch_list]
            )[mask]
            energy_pred = outputs["energy"][mask]
            energy_target, energy_pred = denormalize_tensors(
                batch_list,
                "energy",
                (energy_target, energy_pred),
                mask[:, task_idx],
            )

            energy_mae = per_task_energy_maes[task_idx](
                energy_pred, energy_target
            )
            metrics = self.evaluator.update(
                f"task_{task_idx}_energy_mae", energy_mae.item(), metrics
            )

            forces_target = torch.cat(
                [batch.forces_onehot.to(self.device) for batch in batch_list]
            )[node_mask]
            forces_pred = outputs["forces"][node_mask]
            forces_target, forces_pred = denormalize_tensors(
                batch_list,
                "forces",
                (forces_target, forces_pred),
                node_mask[:, task_idx],
            )

            forces_mae = per_task_force_maes[task_idx](
                forces_pred, forces_target
            )
            metrics = self.evaluator.update(
                f"task_{task_idx}_forces_mae", forces_mae.item(), metrics
            )

        if is_eval:
            # If we're evaluating, then we need to return any intermediate computations.
            # We just aggregate everything for the final metric.
            return {}

        return metrics

    @override
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        metrics = super().validate(split, disable_tqdm)

        # For the per-task metrics, we compute and return the state here
        # so that we can report the final metric.
        for task_idx in range(self.num_tasks):
            energy_mae = self.val_per_task_energy_maes[task_idx]
            metrics = self.evaluator.update(
                f"task_{task_idx}_energy_mae",
                energy_mae.compute().item(),
                metrics,
            )
            energy_mae.reset()

            force_mae = self.val_per_task_force_maes[task_idx]
            metrics = self.evaluator.update(
                f"task_{task_idx}_forces_mae",
                force_mae.compute().item(),
                metrics,
            )
            force_mae.reset()

        return metrics

    @override
    def _compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch_list: List[Batch],
        evaluator: Evaluator,
        metrics: Dict[str, Any] = {},
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

        with denormalize_context(batch_list, aggregated_outputs) as (
            denormed_batch_list,
            denormed_aggregated_outputs,
        ):
            metrics = super()._compute_metrics(
                denormed_aggregated_outputs.copy(),
                denormed_batch_list,
                evaluator,
                metrics,
            )

        # Now, compute the per-task metrics
        _ = metrics.update(self._compute_per_task_metrics(outputs, batch_list))

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
        num_tasks = self.num_tasks
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
                    value.ndim == 2 and value.shape[1] == num_tasks
                ), f"System output {value.shape=} should be (bsz, {num_tasks=})"
            elif system == "atom":
                # Shape should be (n_atoms, n_tasks, 3)
                assert (
                    value.ndim == 3 and value.shape[1] == num_tasks
                ), f"Atom output {value.shape=} should be (n_atoms, {num_tasks=}, 3)"
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
