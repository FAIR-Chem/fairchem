import copy
import sys
from argparse import Namespace
from pathlib import Path

import torch
from minydra import resolved_args
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.common.timer import Times
from ocpmodels.common.utils import (
    build_config,
    move_lmdb_data_to_slurm_tmpdir,
    resolve,
    setup_imports,
)
from ocpmodels.trainers.single_trainer import SingleTrainer

if __name__ == "__main__":
    args = resolved_args(
        defaults={
            "base_path": "$SCRATCH/ocp/runs",
            "n_loops": 1,
            "others": "",
            "job_ids": "",
        },
    ).pretty_print()
    base = resolve(args.base_path)
    job_ids = [j.strip() for j in str(args.job_ids).split(",")]
    paths = [Path(base) / j for j in job_ids if j] + [
        resolve(p.strip()) for p in args.others.split(",")
    ]
    run_dir = resolve("$SCRATCH/ocp/inference_time")

    setup_imports()

    torch.set_grad_enabled(False)

    conf_args = [
        Namespace(
            restart_from_dir=str(p),
            continue_from_dir=None,
            keep_orion_config=False,
            run_dir=run_dir / "-".join(job_ids),
            num_nodes=1,
            num_gpus=1,
        )
        for p in paths
    ]
    configs = [
        build_config(ca, [], silent=True)
        for ca in tqdm(conf_args, desc="Loading configs".ljust(40))
    ]
    configs = [(l, config) for config in configs for l in range(args.n_loops)]
    names = [
        f'{config["restart_from_dir"].name}-{config["config"]}' for _, config in configs
    ]

    times = {}

    for k, (l, config) in enumerate(
        tqdm(
            configs,
            desc=f"Timing {args.n_loops}x{len(conf_args)}={len(configs)} configs".ljust(
                40
            ),
        )
    ):
        config["logger"] = "dummy"
        config["silent"] = True

        od = copy.deepcopy(config["dataset"])
        for split in od:
            if split != "default_val" and split != config["dataset"]["default_val"]:
                del config["dataset"][split]
        config = move_lmdb_data_to_slurm_tmpdir(config)
        for split in od:
            if split != "default_val" and split != config["dataset"]["default_val"]:
                config["dataset"][split] = od[split]

        if l == 0:
            trainer = SingleTrainer(**config)
            timer = Times(gpu=True)

        name = names[k]

        for i, b in enumerate(
            tqdm(
                trainer.loaders[trainer.config["dataset"]["default_val"]],
                desc=f"{name} (loop {l+1}/{args.n_loops})".ljust(40),
                leave=False,
            )
        ):
            with torch.cuda.amp.autocast(enabled=trainer.scaler is not None):
                with timer.next("forward"):
                    _ = trainer.model_forward(b, mode="inference")

        if l == args.n_loops - 1:
            mean, std = timer.prepare_for_logging(
                map_func=lambda t: t / trainer.config["optim"]["batch_size"]
            )
            times[name] = mean["forward"]

    print(
        " •  "
        + "\n •  ".join(
            f"{k}: {v:.6f} s / sample = {1/v:.2f} samples / s" for k, v in times.items()
        )
    )
