"""
Exploration file to get a `batch` in-memory and play around with it.
Use it in notebooks or ipython console

$ ipython
...
In [1]: run get_data_sample.py
Out[1]: ...

In [2]: print(batch)

"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa: F401
from minydra import resolved_args
from tqdm import tqdm

from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import build_config, setup_imports, setup_logging
from ocpmodels.preprocessing import remove_tag0_nodes

if __name__ == "__main__":

    opts = resolved_args()

    sys.argv[1:] = ["--mode=train", "--config=configs/is2re/10k/schnet/schnet.yml"]
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    config["optim"]["num_workers"] = 4
    config["optim"]["batch_size"] = 2
    config["logger"] = "dummy"

    if opts.victor_local:
        config["dataset"][0]["src"] = "data/is2re/10k/train/data.lmdb"

    setup_imports()
    trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
        task=config["task"],
        model_attributes=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        timestamp_id=config.get("timestamp_id", None),
        run_dir=config.get("run_dir", "./"),
        is_debug=config.get("is_debug", False),
        print_every=config.get("print_every", 100),
        seed=config.get("seed", 0),
        logger=config.get("logger", "wandb"),
        local_rank=config["local_rank"],
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
        slurm=config.get("slurm", {}),
        new_gnn=config.get("new_gnn", True),
        data_split=config.get("data_split", None),
        note=config.get("note", ""),
    )

    task = registry.get_task_class(config["mode"])(config)
    task.setup(trainer)

    if opts.no_tag_0 is None:
        for batch in trainer.train_loader:
            b = batch[0]
            rewired = remove_tag0_nodes(b)
            break

    if opts.plot_tags is not None:
        tags = {
            0: [],
            1: [],
            2: [],
        }
        for batch in tqdm(trainer.train_loader):
            for b in batch:
                for t in tags:
                    tags[t].append((b.tags == t).sum().item())

        x = np.arange(len(tags[0]))
        ys = [np.array(tags[t]) for t in range(3)]
        z = np.zeros(len(x))
        fig = plt.figure(num=1)
        ax = fig.add_subplot(111)
        colors = {
            0: "b",
            1: "y",
            2: "g",
        }
        for t in tags:
            ax.plot(x, ys[t], color=colors[t], lw=1, label=f"tag {t}")
        for t in tags:
            ax.fill_between(
                x, ys[t], where=ys[t] > z, color=colors[t], interpolate=True
            )
        plt.legend()
        plt.savefig("tags_dist.png", dpi=150)
