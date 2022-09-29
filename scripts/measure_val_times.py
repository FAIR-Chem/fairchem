"""
Python script to time a list of OCP trainer configurations
|
use as:
|
$ pyhton scripts/measure_val_times.py n_runs=5 output_filename="times_n_runs_5.json"
|
Re-running with the same output_filename will resume timing, ignoring already timed
configs. A config is considered timed after all its runs are completed.
"""
import json
import os
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

if Path.cwd().name == "scripts":
    sys.path.append("..")

from time import time

from minydra import resolved_args
from tqdm import tqdm

from ocpmodels.common.utils import make_script_trainer

try:
    import ipdb  # noqa: F401

    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
except:  # noqa: E722
    pass


def print_time_stats(times_dict: dict, keylen: int = 60) -> None:
    """
    Prints a dictionnary of time statistics

    Args:
        times_dict (dict): the time to print, expects "mean" and "std" sub keys
        keylen (int, optional): the length of the keys to have uniform spacing.
            Defaults to 55.
    """
    for note, s in times_dict.items():
        n = note + "  " + "." * (keylen - len(note))
        print(f"• {n} {s['mean']:8.4f}s +/- {s['std']:.4f}s")


def save_and_print(
    output_json: Path,
    times: dict,
    rewiring_times: dict,
    configs: dict,
    TRAINER_CONF_OVERRIDES: dict,
) -> None:
    """
    Checkpoints the timing state and prints a summary of its keys

    Args:
        output_json (Path): Where to write the checkpoint
        times (dict): Trainer timings
        rewiring_times (dict): Graph rewiring timings
        configs (dict): The trainer configs we're timing
        TRAINER_CONF_OVERRIDES (dict): The trainer's overriding args (like
            batch_size etc.)
    """

    val_times = {
        note: [sum(epoch) for epoch in epochs] for note, epochs in times.items()
    }
    epoch_rewiring_times = {
        note: [sum(epoch) for epoch in epochs]
        for note, epochs in rewiring_times.items()
    }

    stats = {
        note: {
            "mean": np.mean(epoch),
            "std": np.std(epoch),
        }
        for note, epoch in val_times.items()
    }

    rewiring_stats = {
        note: {
            "mean": np.mean(epoch),
            "std": np.std(epoch),
        }
        for note, epoch in epoch_rewiring_times.items()
    }

    print("\nSTATS:")
    print_time_stats(stats)
    print("\nREWIRINGS")
    print_time_stats(rewiring_stats)
    print()
    print("-" * 82)
    print()

    with output_json.open("w") as f:
        out_dict = {
            "configs": configs,
            "TRAINER_CONF_OVERRIDES": TRAINER_CONF_OVERRIDES,
            "times": times,
            "val_times": val_times,
            "stats": stats,
            "rewiring_times": rewiring_times,
            "epoch_rewiring_times": epoch_rewiring_times,
            "rewiring_stats": rewiring_stats,
        }
        json.dump(out_dict, f)


TRAINER_CONF_OVERRIDES = {
    "optim": {
        "num_workers": 6,
        "eval_batch_size": 64,
    },
    "logger": "dummy",
}


ALL_CONFIGS = [
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
        ],
        "overrides": {**TRAINER_CONF_OVERRIDES, "note": "schnet Baseline"},
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Baseline",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
        ],
        "overrides": {**TRAINER_CONF_OVERRIDES, "note": "forcenet Baseline"},
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.graph_rewiring=remove-tag-0",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Rewiring: Remove Tag-0",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.graph_rewiring=remove-tag-0",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Rewiring: Remove Tag-0",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.graph_rewiring=remove-tag-0",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Rewiring: Remove Tag-0",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.graph_rewiring=one-supernode-per-graph",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Rewiring: 1 SN per Graph",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.graph_rewiring=one-supernode-per-graph",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Rewiring: 1 SN per Graph",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.graph_rewiring=one-supernode-per-graph",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Rewiring: 1 SN per Graph",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.graph_rewiring=one-supernode-per-atom-type",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Rewiring: 1 SN per Atom Type",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.graph_rewiring=one-supernode-per-atom-type",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Rewiring: 1 SN per Atom Type",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.graph_rewiring=one-supernode-per-atom-type",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Rewiring: 1 SN per Atom Type",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.graph_rewiring=one-supernode-per-atom-type-dist",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Rewiring: 1 SN per Atom Type Dist",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.tag_hidden_channels=32",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Tag Hidden Channels 32",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.tag_hidden_channels=32",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Tag Hidden Channels 32",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.tag_hidden_channels=32",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Tag Hidden Channels 32",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.tag_hidden_channels=64",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Tag Hidden Channels 64",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.tag_hidden_channels=64",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Tag Hidden Channels 64",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.tag_hidden_channels=64",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Tag Hidden Channels 64",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.phys_embeds=true",
        ],
        "overrides": {**TRAINER_CONF_OVERRIDES, "note": "schnet Phys Embeds Fixed"},
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.phys_embeds=true",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Phys Embeds Fixed",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.phys_embeds=true",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Phys Embeds Fixed",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.phys_embeds=true",
            "--model.phys_hidden_channels=32",
        ],
        "overrides": {**TRAINER_CONF_OVERRIDES, "note": "schnet"},
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.phys_embeds=true",
            "--model.phys_hidden_channels=32",
        ],
        "overrides": {**TRAINER_CONF_OVERRIDES, "note": "dimenet_plus_plus"},
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.phys_embeds=true",
            "--model.phys_hidden_channels=32",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Phys Embeds Learned",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.pg_hidden_channels=32",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Learned Period & Group",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.pg_hidden_channels=32",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Learned Period & Group",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.pg_hidden_channels=32",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Learned Period & Group",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.phys_embeds=true",
            "--model.phys_hidden_channels=32",
            "--model.pg_hidden_channels=32",
            "--model.tag_hidden_channels=64",
        ],
        "overrides": {**TRAINER_CONF_OVERRIDES, "note": "schnet All embeddings"},
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.phys_embeds=true",
            "--model.phys_hidden_channels=32",
            "--model.pg_hidden_channels=32",
            "--model.tag_hidden_channels=64",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus All embeddings",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.phys_embeds=true",
            "--model.phys_hidden_channels=32",
            "--model.pg_hidden_channels=32",
            "--model.tag_hidden_channels=64",
        ],
        "overrides": {**TRAINER_CONF_OVERRIDES, "note": "forcenet All embeddings"},
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.energy_head=weighted-av-initial-embeds",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Energy Head: Weighted Average (initial)",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.energy_head=weighted-av-initial-embeds",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Energy Head: Weighted Average (initial)",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.energy_head=weighted-av-initial-embeds",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Energy Head: Weighted Average (initial)",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.energy_head=weighted-av-final-embeds",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Energy Head: Weighted Average (final)",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.energy_head=weighted-av-final-embeds",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Energy Head: Weighted Average (final)",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.energy_head=weighted-av-final-embeds",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Energy Head: Weighted Average (final)",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.energy_head=pooling",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Energy Head: Pooling",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.energy_head=pooling",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Energy Head: Pooling",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.energy_head=pooling",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Energy Head: Pooling",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            "--model.energy_head=graclus",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "schnet Energy Head: Graclus",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            "--model.energy_head=graclus",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "dimenet_plus_plus Energy Head: Graclus",
        },
    },
    {
        "str_args": [
            "--mode=train",
            "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            "--model.energy_head=graclus",
        ],
        "overrides": {
            **TRAINER_CONF_OVERRIDES,
            "note": "forcenet Energy Head: Graclus",
        },
    },
]


if __name__ == "__main__":

    # parse command-line arguments
    args = resolved_args(
        defaults={
            "n_runs": 1,  # (int) runs for the same trainer
            "max_confs": -1,  # (int) crop the number of confs to test
            "max_batchs": -1,  # (int) crop the number of batchs to test
            "conf_ids": [],  # (list[int]) list of conf ids to measure. Defaults to all if empty
            "output_filename": "measured_times.json",  # (str) output file name in data/times/
            "ignore_confs": [],  # (list[int]) list of config indices to ignore
            "overwrite": False,  # (bool) overwrite existing file
            "dryrun": False,  # (bool) run things but don't load from/save to file
            "ignores": [],  # (list[int]) list of config ids to ignore
        }
    ).pretty_print()

    # safe-guard all configs
    configs = deepcopy(ALL_CONFIGS)

    if args.max_confs > 0:
        # crop the number of confs to process, for debugging purposes
        configs = configs[: args.max_confs]
    elif args.conf_ids:
        # select a single conf
        if args.max_confs > 0:
            print(
                "\nWARNING: `conf_ids` has precedence over `max_confs`",
                "which will be ignored\n",
            )
        configs = [configs[c] for c in args.conf_ids]

    # initiate the time and rewiring time data stores
    times = {c["overrides"]["note"]: [[] for _ in range(args.n_runs)] for c in configs}
    rewiring_times = {}

    # disable gradients for validation inference
    torch.set_grad_enabled(False)

    # create the output folder and file
    output_dir = Path(__file__).resolve().parent.parent / "data" / "times"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_json = output_dir / args.output_filename

    # existing_confs is initially empty
    existing_confs = set()
    if output_json.exists() and not args.overwrite and not args.dryrun:
        # there's an output file: use it to load existing timings
        with output_json.open("r") as f:
            existing_times = json.load(f)

        # overwrite times and rewiring_times
        times = existing_times["times"]
        rewiring_times = existing_times["rewiring_times"]
        # update existing_confs
        existing_confs = set(k for k in times if any(times[k]))

    # for each trainer configuration
    for c, config in enumerate(configs):
        note = config["overrides"]["note"]
        if note in existing_confs:
            # config has already been timed: ignore
            print(f"🤠 Data for `{note}` already exists. Skipping.")
            continue
        if c in args.ignores:
            print(f"🙈 config {c} is ignored from the command-line.")
            continue

        # create trainer in eval modefor this config
        trainer = make_script_trainer(**config, verbose=False)
        trainer.model.eval()

        # for each run (to have stats)
        for r in range(args.n_runs):

            # print status
            print(
                f"\n\n🔄 Timing {note} ({c+1}/{len(configs)})",
                f"-> Run {r+1}/{args.n_runs}\n\n",
            )

            # for each batch in the val-id dataset
            for b, batch in enumerate(tqdm(trainer.val_loader, desc=note)):
                # time the forward pass
                t = time()
                with torch.cuda.amp.autocast(enabled=trainer.scaler is not None):
                    _ = trainer._forward(batch)
                forward_duration = time() - t

                if trainer.model.module.graph_rewiring:
                    # remove the rewiring time which really should be done in
                    # the data loader
                    forward_duration -= trainer.model.module.rewiring_time
                    # store rewiring time
                    if note not in rewiring_times:
                        rewiring_times[note] = []
                    if len(rewiring_times[note]) == r:
                        rewiring_times[note].append([])
                    rewiring_times[note][r].append(trainer.model.module.rewiring_time)

                # store forward time
                times[note][r].append(forward_duration)
                if args.max_batchs > 0 and b > args.max_batchs - 1:
                    # break for debugs
                    break

        # all runs have been completed for this config: checkpoint timing state
        if not args.dryrun:
            save_and_print(
                output_json, times, rewiring_times, configs, TRAINER_CONF_OVERRIDES
            )
