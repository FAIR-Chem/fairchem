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
    def load_datasets(self):
        super().load_datasets()

        for attr_name in ["train_dataset", "val_dataset", "test_dataset"]:
            dataset = getattr(self, attr_name, None)
            if dataset is None:
                continue

            dataset.transform = Data.from_torch_geometric_data

    def get_dataloader(self, dataset, sampler):
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
        return self.config["model_attributes"].get(
            "regress_positions",
            self.config["model_attributes"].get("regress_forces", False),
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._pos_enabled:
            self._pos_evaluator = Evaluator(task="is2rs")
            self._pos_evaluator.task_attributes["is2rs"] = ["position"]
            self._pos_evaluator.task_metrics["is2rs"] = [
                "positions_mae",
                "positions_mse",
            ]

            if self.normalizer.get("normalize_positions", False):
                self.normalizers["position"] = Normalizer(
                    mean=self.normalizer["positions_mean"],
                    std=self.normalizer["positions_std"],
                    device=self.device,
                )

    @property
    def _energy_multiplier(self) -> float:
        return self.config["optim"].get("energy_coefficient", 1.0)

    @property
    def _force_multiplier(self) -> float:
        weight = self.config["optim"].get("force_coefficient", 12.5)
        config = self.config["optim"].get("force_coefficient_decay", None)
        if config and config.get("enabled", False):
            weight_range = max(0.0, weight - config["min_weight"])
            weight -= weight_range * min(
                1.0, self.step / config["total_steps"]
            )
        return weight

    def _forward(self, batch_list):
        out = {}
        out["energy"], out["position"], out["position_mask"] = self.model(
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

    def _get_position_target(self, batch_list, *, norm: bool):
        pos = torch.cat(
            [batch.deltapos.to(self.device) for batch in batch_list], dim=0
        )
        if norm and self.normalizers.get("normalize_positions", False):
            pos = self.normalizers["position"].norm(pos)
        return pos

    def _compute_loss(self, out, batch_list):
        output, node_output, node_target_mask = (
            out["energy"],
            out["position"],
            out["position_mask"],
        )

        relaxed_energy = self._get_energy_target(batch_list, norm=True)
        eng_loss = self.loss_fn["energy"](
            output.float().view(-1), relaxed_energy
        )

        deltapos = self._get_position_target(batch_list, norm=True)
        node_loss = self.loss_fn["force"](
            _mask(node_output, mask=node_target_mask).float(),
            _mask(deltapos, mask=node_target_mask),
        )

        return (
            self._energy_multiplier * eng_loss
            + self._force_multiplier * node_loss
        )

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )

        target = {
            "energy": self._get_energy_target(batch_list, norm=False),
            "position": self._get_position_target(batch_list, norm=False),
            "natoms": natoms,
        }

        out["position"] = _mask(out["position"], mask=out["position_mask"])
        target["position"] = _mask(
            target["position"], mask=out["position_mask"]
        )

        out["natoms"] = natoms

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])

        if self.normalizers.get("normalize_positions", False):
            out["position"] = self.normalizers["position"].denorm(
                out["position"]
            )

        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        if self._pos_enabled:
            metrics = self._pos_evaluator.eval(
                out,
                target,
                prev_metrics=metrics,
            )
        metrics.update(
            {
                "energy_coefficient": dict(
                    metric=self._energy_multiplier,
                    total=self._energy_multiplier,
                    numel=1,
                ),
                "position_coefficient": dict(
                    metric=self._force_multiplier,
                    total=self._force_multiplier,
                    numel=1,
                ),
            }
        )
        return metrics
