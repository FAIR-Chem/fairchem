from logging import getLogger
from typing import cast

import torch
from einops import reduce
from torch_geometric.data import Batch
from typing_extensions import override

from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.common.registry import registry
from ocpmodels.common.typed_config import TypeAdapter

from ..ocp_trainer import OCPTrainer
from .dataset import DatasetConfig, create_datasets
from .loss import LossFnsConfig, create_losses

log = getLogger(__name__)


@registry.register_trainer("mt_trainer")
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
        super().__init__(
            task,
            model,
            outputs,
            DatasetConfig.from_dict(
                dataset
            ),  # HACK: wrap it in a class so it doesn't get registered as a dict
            optimizer,
            TypeAdapter(LossFnsConfig).validate_python(loss_fns),
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

    @property
    def loss_fns_config(self):
        loss_fns_config = self.config["loss_fns"]
        return cast(LossFnsConfig, loss_fns_config)

    @override
    def load_loss(self) -> None:
        self.loss_fns = list(create_losses(self.loss_fns_config))

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
                    value.ndim == 2 and value.shape[-1] == num_tasks
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
        self._validate_mt_outputs(outputs)
        return outputs
