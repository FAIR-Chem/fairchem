from pathlib import Path
import sys
import numpy as np

if Path.cwd().name == "scripts":
    sys.path.append("..")

from ocpmodels.common.utils import make_trainer

import ocpmodels.preprocessing as ocpreproc
from tqdm import tqdm
from time import time

if __name__ == "__main__":

    trainer_conf_overrides = {
        "optim": {
            "num_workers": 4,
            "batch_size": 64,
        },
        "logger": "dummy",
    }

    trainer = make_trainer(
        str_args=["--mode=train", "--config=configs/is2re/all/schnet/new_schnet.yml"],
        overrides=trainer_conf_overrides,
    )

    times = {
        "one_supernode_per_atom_type": [],
        "one_supernode_per_atom_type_dist": [],
        "one_supernode_per_graph": [],
        "remove_tag0_nodes": [],
    }

    for func_name in times:
        func_to_time = getattr(ocpreproc, func_name)
        for batch in tqdm(trainer.val_loader, desc=func_name):
            t = time()
            func_to_time(batch[0])
            times[func_name].append(time() - t)

    print()
    m = max([len(_) for _ in times]) + 1

    for k, ts in times.items():
        print(f"{k:{m}}: {np.mean(ts):.3f} (+/- {np.std(ts):.3f})")
