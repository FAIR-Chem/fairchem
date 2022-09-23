from pathlib import Path
import sys
import numpy as np
import torch

if Path.cwd().name == "scripts":
    sys.path.append("..")

from ocpmodels.common.utils import make_trainer

from tqdm import tqdm
from time import time
from minydra import resolved_args

if __name__ == "__main__":

    args = resolved_args().pretty_print()

    n_runs = args.runs or 1

    trainer_conf_overrides = {
        "optim": {
            "num_workers": 6,
            "eval_batch_size": 64,
        },
        "logger": "dummy",
    }

    configs = [
        {
            "str_args": [
                "--mode=train",
                "--config=configs/is2re/all/schnet/new_schnet.yml",
                "--model.graph_rewiring='remove-tag-0'",
            ],
            "overrides": {
                **trainer_conf_overrides,
                "note": "schnet - tag0",
            },
        }
    ]

    times = {c["overrides"]["note"]: [[] for _ in range(n_runs)] for c in configs}

    torch.set_grad_enabled(False)

    for config in configs:
        for r in range(n_runs):
            trainer = make_trainer(**config)
            trainer.model.eval()
            note = config["overrides"]["note"]
            for batch in tqdm(trainer.val_loader, desc=note):
                t = time()
                with torch.cuda.amp.autocast(enabled=trainer.scaler is not None):
                    _ = trainer._forward(batch)
                forward_duration = time() - t
                if trainer.model.module.graph_rewiring:
                    forward_duration -= trainer.model.module.rewiring_time
                times[note][r].append(forward_duration)

    val_times = {
        note: [sum(epoch) for epoch in epochs] for note, epochs in times.items()
    }

    print(val_times)

    stats = {
        note: {
            "mean": np.mean(epoch),
            "std": np.std(epoch),
        }
        for note, epoch in val_times.items()
    }

    print("\nSTATS:")
    print(stats)
