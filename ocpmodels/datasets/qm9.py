from pathlib import Path

import time
import json
import torch
from torch_geometric.datasets import QM9

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import ROOT


@registry.register_dataset("qm9")
class QM9Dataset(QM9):
    """
    Original dataset:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html?highlight=qm9#torch_geometric.datasets.QM9 # noqa: E501
    """

    def __init__(
        self,
        config={
            "src": "/network/projects/ocp/qm9",
            "target": 7,
            "seed": 123,
            "normalize_labels": True,
            "target_mean": -11178.966796875,
            "target_std": 1085.5787353515625,
            "indices": {"start": 0, "end": 110000},
        },
        transform=None,
    ):
        self.root = Path(config["src"])
        self.config = config
        assert self.root.exists(), f"QM9 dataset not found in {config['src']}"
        super().__init__(str(self.root))
        self.base_length = super().__len__()  # full dataset length
        self.target = config["target"]  # index of target to predict

        # `._transform` not to conflict with built-in `.transform` which
        # would break the super().__getitem__ call
        self._transform = transform

        # randomize samples except if seed is < 0

        if config["seed"] >= 0:
            g = torch.Generator()
            g.manual_seed(config["seed"])
            self.perm = torch.randperm(self.base_length, generator=g)
        else:
            self.perm = torch.arange(self.base_length)

        start = int(config["indices"]["start"])
        end = int(config["indices"]["end"])

        if start == 0:
            if end == -1:
                self.samples = self.perm
            else:
                self.samples = self.perm[:end]
        elif end == -1:
            self.samples = self.perm[start:]
        else:
            self.samples = self.perm[start:end]

        self.lse_shifts = None
        if self.config.get("lse_shift"):
            self.lse_shifts = torch.tensor(
                json.loads(
                    (
                        ROOT
                        / "configs"
                        / "models"
                        / "qm9-metadata"
                        / "lse-shifts-pre-attr.json"
                    ).read_text()
                )
            )

    def close_db(self):
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t0 = time.time_ns()
        data = super().__getitem__(self.samples[idx])
        data.y = data.y[0, self.target]
        data.natoms = len(data.pos)
        data.atomic_numbers = data.z
        data.cell_offsets = torch.zeros((data.edge_index.shape[1], 3))
        del data.z
        data.tags = torch.full((data.natoms,), -1, dtype=torch.long)

        if self.lse_shifts is not None:
            data.lse_shift = self.lse_shifts[self.target][data.atomic_numbers].sum()
            data.y_unshifted = data.y
            data.y = data.y - data.lse_shift

        t1 = time.time_ns()
        if self._transform is not None:
            data = self._transform(data)
        t2 = time.time_ns()

        load_time = (t1 - t0) * 1e-9  # time in s
        transform_time = (t2 - t1) * 1e-9  # time in s
        total_get_time = (t2 - t0) * 1e-9  # time in s

        data.load_time = load_time
        data.transform_time = transform_time
        data.total_get_time = total_get_time

        return data


if __name__ == "__main__":
    from ocpmodels.datasets.qm9 import QM9Dataset as QMD

    train_set = QMD(
        {
            "src": "/network/projects/ocp/qm9/",  # where's the data
            "seed": 123,  # random seed for shuffling
            "ratio": {"start": 0, "end": 0.7},  # what fraction of the data to use
            "target": 7,  # which property to predict
        }
    )
    print(train_set[0])
