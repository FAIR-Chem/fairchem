from pathlib import Path

import time

import torch
from torch_geometric.datasets import QM9

from ocpmodels.common.registry import registry


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
            "ratio": {"start": 0, "end": 0.75},
        },
        transform=None,
    ):
        self.root = Path(config["src"])
        self.config = config
        assert self.root.exists(), f"QM9 dataset not found in {config['src']}"
        super().__init__(str(self.root))
        self.base_length = super().__len__()
        self.target = config["target"]
        # `._transform` not to conflict with built-in `.transform` which
        # would break the super().__getitem__ call
        self._transform = transform
        g = torch.Generator()
        g.manual_seed(config["seed"])
        perm = torch.randperm(self.base_length, generator=g)
        if config["ratio"]["start"] == 0:
            self.samples = perm[: int(config["ratio"]["end"] * self.base_length)]
        else:
            start = int(config["ratio"]["start"] * self.base_length)
            end = int(config["ratio"]["end"] * self.base_length)
            self.samples = perm[start:end]

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
