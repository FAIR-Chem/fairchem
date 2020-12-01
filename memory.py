from pathlib import Path
import numpy as np
import sys
from ocpmodels.common.utils import get_pbc_distances
import torch
from ocpmodels.common.data_parallel import OCPDataParallel, ParallelCollater
from torch.utils.data.dataloader import DataLoader
from ocpmodels.datasets.trajectory_lmdb import TrajectoryLmdbDataset
from ocpmodels.models.dimenet_plus_plus import DimeNetPlusPlusWrap
from ocpmodels.models import DimeNetPlusPlus
import yaml
from matplotlib import pyplot as plt


def model_forward(model, batch):
    mem1 = torch.cuda.max_memory_allocated()
    with torch.cuda.amp.autocast(enabled=True):
        energy, forces = model(batch)
    mem2 = torch.cuda.max_memory_allocated()
    mem = (mem2 - mem1) / 1e9
    print("Memory:", mem)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return energy, forces


def main():
    dataset = TrajectoryLmdbDataset({
        "src": "data/s2ef/200k/train/",
        "dynamic_batching": True,
        "atomsdir": "data/s2ef/200k/train.atoms",
    })
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=ParallelCollater(1, False),
        num_workers=10,
        pin_memory=True,
    )
    model_cfg = yaml.safe_load(open("configs/sweeps/dimenetpp_tiny_200k_emb64.yml"))["model"]
    del model_cfg["name"]

    model = DimeNetPlusPlusWrap(None, bond_feat_dim=50, num_targets=1, **model_cfg).cuda()
    print(model)
    print("#Params:", model.num_params)
    model = OCPDataParallel(model, output_device=0, num_gpus=1)

    for i, batch in enumerate(loader):
        print(batch)
        energy, forces = model_forward(model, batch)
        if i == 10:
            break


if __name__ == "__main__":
    main()
