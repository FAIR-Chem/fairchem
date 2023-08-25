from collections import defaultdict
from logging import getLogger
from typing import Dict, Literal, Union, cast

import torch
from einops import reduce
from torch.nn.parallel.distributed import DistributedDataParallel
from torch_geometric.data import Batch
from typing_extensions import Annotated, override

from ocpmodels.common import distutils
from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollater
from ocpmodels.common.registry import registry
from ocpmodels.common.typed_config import Field, TypeAdapter, TypedConfig
from ocpmodels.modules.evaluator import Evaluator

from ..ocp_trainer import OCPTrainer
from .dataset import DatasetConfig, create_datasets
from .loss import LossFnsConfig, create_losses

log = getLogger(__name__)


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

    @override
    def __post_init__(self):
        super().__post_init__()

        if self.irrep_dim != 1:
            raise NotImplementedError(
                f"Only irrep_dim=1 is supported for the MT trainer."
            )


OutputHeadConfig = Annotated[
    Union[SystemLevelOutputHeadConfig, AtomLevelOutputHeadConfig],
    Field(discriminator="level"),
]

OutputsConfig = Annotated[Dict[str, OutputHeadConfig], Field()]


@registry.register_trainer("mt")
class MTTrainer(OCPTrainer):
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
        name="ocp",
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
            slurm,
            noddp,
            name,
        )

    @override
    def load_task(self):
        # We don't use this way of normalizing.
        self.normalizers = {}

        outputs_config = TypeAdapter(OutputsConfig).validate_python(
            self.config["outputs"]
        )
        self.multi_task_targets = defaultdict[str, list[str]](lambda: [])
        self.per_task_output_targets = {}
        self.output_targets = {}
        for target_name, target_config in outputs_config.items():
            target_config_dict = target_config.to_dict()
            self.output_targets[target_name] = target_config_dict.copy()
            for task_idx in range(len(self.dataset_config.datasets)):
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

    @override
    def load_model(self) -> None:
        # Build model
        if distutils.is_master():
            log.info(f"Loading model: {self.config['model']}")

        self.model = registry.get_model_class(self.config["model"])(
            self.per_task_output_targets,
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
        ) = create_datasets(self.dataset_config)

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

    @override
    def load_loss(self) -> None:
        loss_fns_config = TypeAdapter(LossFnsConfig).validate_python(
            self.config["loss_fns"]
        )
        self.loss_fns = list(create_losses(loss_fns_config))

    @override
    def _compute_loss(
        self,
        out: dict[str, torch.Tensor],
        batch_list: list[Batch],
    ):
        batch_idx = torch.cat(
            [batch.batch.to(self.device) for batch in batch_list]
        )  # (n_atoms,)

        fixed = torch.cat(
            [batch.fixed.to(self.device) for batch in batch_list]
        )
        free_mask = fixed == 0  # (n_atoms,)

        batch_task_idx = torch.cat(
            [batch.task_idx.to(self.device) for batch in batch_list], dim=0
        )  # (bsz,)
        node_task_idx = batch_task_idx[batch_idx]  # (n_atoms,)

        losses: list[torch.Tensor] = []
        for target_name, loss_info in self.loss_fns:
            target = torch.cat(
                [batch[target_name].to(self.device) for batch in batch_list],
                dim=0,
            )
            pred = out[target_name]
            mask = torch.ones_like(target, dtype=torch.bool)

            level = self.output_targets[target_name].get("level", "system")

            # For each task, we want to only consider the loss for that task's output head.
            task_idx = loss_info.get("task_idx", None)
            if task_idx is not None:
                if level == "atom":
                    mask = mask & (node_task_idx == task_idx)  # (n_atoms,)
                elif level == "system":
                    mask = mask & (batch_task_idx == task_idx)  # (bsz,)
                else:
                    raise NotImplementedError(
                        f"Level {level} not implemented."
                    )

            if level == "atom" and self.output_targets[target_name].get(
                "train_on_free_atoms", True
            ):
                mask = mask & free_mask

            normalizer = self.normalizers.get(target_name, False)
            if normalizer:
                target = normalizer.norm(target)

            # Compute the loss
            loss = loss_info["fn"](
                pred, target
            )  # (bsz,) or (n_atoms,) or (natoms, 3)

            # Mask out invalid values
            loss = torch.where(mask, loss, torch.zeros_like(loss))

            # Apply the coefficient
            coeff = loss_info.get("coefficient", 1.0)
            loss = coeff * loss

            # Apply the reduction
            reduction = loss_info.get("reduction", "mean")
            if reduction == "sum":
                loss = reduce(loss, "... -> ", "sum")
            elif reduction == "mean":
                loss = reduce(loss, "b ... -> ...", "sum") / reduce(
                    mask, "b -> ", "sum"
                )
                loss = reduce(loss, "... -> ", "mean")
            else:
                raise NotImplementedError(f"{reduction=} not implemented.")

            losses.append(loss)

        # Sanity check to make sure the compute graph is correct.
        for lc in losses:
            assert hasattr(lc, "grad_fn")

        loss = sum(losses)
        loss = cast(torch.Tensor, loss)
        return loss

    def _merge_mt_outputs(self, outputs: dict[str, torch.Tensor]):
        merged_outputs: dict[str, torch.Tensor] = outputs.copy()

        # Merge the per-task outputs
        for target_name, target_key_list in self.multi_task_targets.items():
            merged_outputs[target_name] = torch.stack(
                [merged_outputs.pop(key) for key in target_key_list], dim=1
            )

        return merged_outputs

    def _validate_mt_outputs(self, outputs: dict[str, torch.Tensor]):
        num_tasks = len(self.dataset_config.datasets)
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
