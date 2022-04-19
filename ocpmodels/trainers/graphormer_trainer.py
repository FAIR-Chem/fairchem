import logging

import torch
from torch.utils.data import DataLoader

from ocpmodels.common.registry import registry
from ocpmodels.models.utils.dense_types import Batch, Data
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers import EnergyTrainer


def _mask(input: torch.Tensor, mask: torch.Tensor):
    masked = input[mask]
    return masked


@registry.register_trainer("graphormer_energy")
class GraphromerEnergyTrainer(EnergyTrainer):
    def load_loss(self):
        loss_positions = self.config["optim"].get("loss_positions", None)
        if loss_positions is not None:
            self.config["optim"]["loss_force"] = loss_positions
        return_value = super().load_loss()
        self.loss_fn["positions"] = self.loss_fn.pop("force")
        return return_value

    def load_datasets(self):
        super().load_datasets()

        # sets dataset transforms to convert from
        # PyG data objects to dense tensors
        for attr_name in ["train_dataset", "val_dataset", "test_dataset"]:
            dataset = getattr(self, attr_name, None)
            if dataset is None:
                continue

            dataset.transform = Data.from_torch_geometric_data

    def get_dataloader(self, dataset, sampler):
        # sets dataloader collate fn for dense tensors
        loader = DataLoader(
            dataset,
            collate_fn=Batch.from_data_list,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_sampler=sampler,
        )
        return loader

    @property
    def _pos_enabled(self):
        return self.config["task"].get("regress_positions", True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._pos_enabled:
            logging.info("Relaxed position auxiliary task training enabled")
            self._pos_evaluator = Evaluator(task="is2rs")
            self._pos_evaluator.task_attributes["is2rs"] = ["positions"]
            self._pos_evaluator.task_metrics["is2rs"] = [
                "positions_mae",
                "positions_mse",
            ]

            if self.normalizer.get("normalize_positions", False):
                self.normalizers["positions"] = Normalizer(
                    mean=self.normalizer["positions_mean"],
                    std=self.normalizer["positions_std"],
                    device=self.device,
                )

    @property
    def _energy_multiplier(self) -> float:
        return self.config["optim"].get("energy_coefficient", 1.0)

    @property
    def _positions_multiplier(self) -> float:
        weight = self.config["optim"].get("positions_coefficient", 12.5)
        config = self.config["optim"].get("positions_coefficient_decay", None)
        if config and config.get("enabled", False):
            weight_range = max(0.0, weight - config["min_weight"])
            weight -= weight_range * min(
                1.0, self.step / config["total_steps"]
            )
        return weight

    def _forward(self, batch_list):
        out = {}
        out["energy"], out["positions"], out["positions_mask"] = self.model(
            batch_list
        )

        if out["energy"].shape[-1] == 1:
            out["energy"] = out["energy"].view(-1)

        return out

    def _get_energy_target(self, batch_list, *, norm: bool):
        relaxed_energy = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )
        relaxed_energy = relaxed_energy.float().squeeze(dim=-1)
        # relaxed_energy = (relaxed_energy - self.e_mean) / self.e_std
        if norm and self.normalizer.get("normalize_labels", False):
            relaxed_energy = self.normalizers["target"].norm(relaxed_energy)
        return relaxed_energy

    def _get_positions_target(self, batch_list, *, norm: bool):
        pos = torch.cat(
            [batch.deltapos.to(self.device) for batch in batch_list], dim=0
        )
        if norm and self.normalizer.get("normalize_positions", False):
            pos = self.normalizers["positions"].norm(pos)
        return pos

    def _compute_loss(self, out, batch_list):
        output, node_output, node_target_mask = (
            out["energy"],
            out["positions"],
            out["positions_mask"],
        )

        relaxed_energy = self._get_energy_target(batch_list, norm=True)
        loss = (
            self.loss_fn["energy"](output.float().view(-1), relaxed_energy)
            * self._energy_multiplier
        )

        if self._pos_enabled:
            deltapos = self._get_positions_target(batch_list, norm=True)
            loss += (
                self.loss_fn["positions"](
                    _mask(node_output, mask=node_target_mask).float(),
                    _mask(deltapos, mask=node_target_mask),
                )
                * self._positions_multiplier
            )

        return loss

    def _compute_pos_metrics(self, out, batch_list, metrics={}):
        if not self._pos_enabled:
            return metrics

        target = {
            "positions": self._get_positions_target(batch_list, norm=False),
        }

        out["positions"] = _mask(out["positions"], mask=out["positions_mask"])
        target["positions"] = _mask(
            target["positions"], mask=out["positions_mask"]
        )

        if self.normalizer.get("normalize_positions", False):
            out["positions"] = self.normalizers["positions"].denorm(
                out["positions"]
            )

        metrics = self._pos_evaluator.eval(
            out,
            target,
            prev_metrics=metrics,
        )
        return metrics

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )
        target = {
            "energy": self._get_energy_target(batch_list, norm=False),
            "natoms": natoms,
        }
        out["natoms"] = natoms

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])

        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        metrics = self._compute_pos_metrics(out, batch_list, metrics)
        metrics.update(
            {
                "energy_coefficient": dict(
                    metric=self._energy_multiplier,
                    total=self._energy_multiplier,
                    numel=1,
                ),
                "positions_coefficient": dict(
                    metric=self._positions_multiplier,
                    total=self._positions_multiplier,
                    numel=1,
                ),
            }
        )
        return metrics
