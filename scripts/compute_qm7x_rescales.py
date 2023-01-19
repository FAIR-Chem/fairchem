import json
import sys
from pathlib import Path

import numpy as np
from mendeleev.fetch import fetch_table
from tqdm import tqdm

sys.path.append(Path(__file__).resolve().parent.parent)

from ocpmodels.common.utils import (
    ROOT,
    base_config,
    move_lmdb_data_to_slurm_tmpdir,
)
from ocpmodels.trainers.single_trainer import SingleTrainer

if __name__ == "__main__":
    config = base_config("schnet-qm7x-all")
    config["cp_data_to_tmpdir"] = True
    config = move_lmdb_data_to_slurm_tmpdir(config)
    trainer = SingleTrainer(**config)

    df = fetch_table("elements")
    HOF = df.set_index("atomic_number")["heat_of_formation"].values
    non_nan_hof_mean = HOF[~np.isnan(HOF)].mean()
    print("non_nan_hof_mean: ", non_nan_hof_mean)  # 353.3106853932584
    HOF[np.isnan(HOF)] = non_nan_hof_mean

    hofs = []

    for batch_list in tqdm(trainer.loaders["train"]):
        hofs += [
            y / HOF[z.astype(int) - 1].sum()
            for y, z in zip(batch_list[0].y, batch_list[0].atNUM)
        ]

    mean = np.mean(hofs)
    std = np.std(hofs)

    (ROOT / "configs" / "models" / "qm7x-metadata" / "hof_rescales.json").write_text(
        json.dumps(
            {
                "mean": float(mean),
                "std": float(std),
                "about": "Statistics for y(=ePBE0+MBD) / sum(HOF) "
                + "where HOF is the heat of formation of each element in the graph."
                + " This is computed over the train set only.",
            }
        )
    )
