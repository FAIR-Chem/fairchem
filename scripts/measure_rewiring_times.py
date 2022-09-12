from pathlib import Path
import sys
import numpy as np

if Path.cwd().name == "scripts":
    sys.path.append("..")

from ocpmodels.common.utils import make_trainer

import ocpmodels.preprocessing as ocpreproc
from tqdm import tqdm
from time import time
from copy import deepcopy
from collections import defaultdict
import matplotlib.pyplot as plt

if __name__ == "__main__":

    trainer_conf_overrides = {
        "optim": {
            "num_workers": 4,
            "eval_batch_size": 1,
        },
        "logger": "dummy",
    }

    trainer = make_trainer(
        str_args=["--mode=train", "--config=configs/is2re/all/schnet/new_schnet.yml"],
        overrides=trainer_conf_overrides,
    )

    base = {
        "per_node": defaultdict(list),
        "per_edge": defaultdict(list),
        "graph": [],
    }

    times = {
        "one_supernode_per_atom_type": deepcopy(base),
        "one_supernode_per_atom_type_dist": deepcopy(base),
        "one_supernode_per_graph": deepcopy(base),
        "remove_tag0_nodes": deepcopy(base),
    }

    for func_name in times:
        func_to_time = getattr(ocpreproc, func_name)
        for batch in tqdm(trainer.val_loader, desc=func_name):
            t = time()
            func_to_time(batch[0])
            duration = time() - t
            times[func_name]["graph"].append(duration)
            times[func_name]["per_node"][batch[0].natoms.item()].append(duration)
            times[func_name]["per_edge"][batch[0].edge_index.shape[1]].append(duration)

    print()
    m = max([len(_) for _ in times]) + 1

    for k, ts in times.items():
        print(f"{k:{m}}: {np.mean(ts):.3f} (+/- {np.std(ts):.3f})")

    plots = {
        k: {kk: {"mean": [], "std": [], "x": []} for kk in v} for k, v in times.items()
    }

    for func_name, modes in times.items():
        for mode, ddict in modes.items():
            if mode == "graph":
                plots[func_name][mode]["mean"] = np.mean(ddict)
                plots[func_name][mode]["std"] = np.std(ddict)
                plots[func_name][mode]["x"] = list(range(len(ddict)))
            else:
                for x, ts in ddict.items():
                    plots[func_name][mode]["mean"].append(np.mean(ts))
                    plots[func_name][mode]["std"].append(np.std(ts))
                    plots[func_name][mode]["x"].append(x)

    labels = [
        "" for _ in range(len(plots["one_supernode_per_atom_type"]["per_node"]["x"]))
    ]

    nrow = 2
    ncol = 2
    fig, axs = plt.subplots(nrows=nrow, ncols=nrow, figsize=(28, 20))
    for i, ax in enumerate(fig.axes):
        mode = list(list(times.values())[0].keys())[i]
        ax.set_ylabel(mode)
        for func_name in times:
            if mode != "graph":
                idx = np.argsort(plots[func_name][mode]["x"])
                x = np.array(plots[func_name][mode]["x"])[idx]
                y = np.array(plots[func_name][mode]["mean"])[idx]
                s = np.array(plots[func_name][mode]["std"])[idx]
                ax.plot(x, y, alpha=0.2)
                ax.fill_between(x, y - s, y + s, alpha=0.3)
            else:
                continue
                plt.plot(
                    plots[func_name][mode]["x"],
                    [plots[func_name][mode]["mean"] for _ in range(len(x))],
                )
        if i > 1:
            break
    fig.tight_layout()
    # ax = plt.gca()
    # # ax.bar_label(rects)
    plt.savefig("test.png")
    plt.clf()
