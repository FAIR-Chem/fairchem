import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.datasets import TrajectoryLmdbDataset


def main():
    seed = 0
    batch_size = 8
    num_workers = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config = {
        "src": "/checkpoint/electrocatalysis/relaxations/features/struct_to_energy_forces/train/200k/"
    }

    train_dataset = TrajectoryLmdbDataset(config)

    data_list_collater = ParallelCollater(1, False)
    collater = data_list_collater
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collater,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    for step, batch in enumerate(train_loader):
        print("============", step, "==========")
        print(batch[0])
        print(batch[0].force)
        print()
        if step > 10:
            break


if __name__ == "__main__":
    main()
