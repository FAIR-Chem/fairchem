from pathlib import Path

import time

import torch
from torch_geometric.datasets import QM9

from ocpmodels.common.registry import registry

Y_MEANS = torch.tensor(
    [
        2.672952651977539,
        75.28118133544922,
        -6.536452770233154,
        0.32204368710517883,
        6.858491897583008,
        1189.4105224609375,
        4.056937217712402,
        -11178.966796875,
        -11178.7353515625,
        -11178.7099609375,
        -11179.875,
        31.620365142822266,
        -76.11600494384766,
        -76.58049011230469,
        -77.01825714111328,
        -70.83665466308594,
        9.966022491455078,
        1.4067283868789673,
        1.1273993253707886,
    ],
    dtype=torch.float32,
)

Y_STDS = torch.tensor(
    [
        1.5034793615341187,
        8.17383098602295,
        0.5977412462234497,
        1.274855375289917,
        1.2841686010360718,
        280.4781494140625,
        0.9017231464385986,
        1085.5787353515625,
        1085.57275390625,
        1085.57275390625,
        1085.5924072265625,
        4.067580699920654,
        10.323753356933594,
        10.415176391601562,
        10.489270210266113,
        9.498342514038086,
        1830.4630126953125,
        1.6008282899856567,
        1.107471227645874,
    ],
    dtype=torch.float32,
)


@registry.register_dataset("qm9")
class QM9Dataset(QM9):
    """
    Original dataset:
    https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html?highlight=qm9#torch_geometric.datasets.QM9 # noqa: E501
    """

    def __init__(self, config, transform=None):
        self.root = Path(config["src"])
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
