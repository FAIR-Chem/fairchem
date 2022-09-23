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

from ocpmodels.common.utils import make_trainer

try:
    import ipdb  # noqa: F401

    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
except:  # noqa: E722
    pass


def print_time_stats(times_dict, keylen=50):
    for note, s in times_dict.items():
        n = note + "  " + "⸱" * (keylen - len(note))
        print(f"• {n} {s['mean']:8.4f}s +/- {s['std']:.4f}s")


def save_and_print(output_json, times, rewiring_times, configs, TRAINER_CONF_OVERRIDES):

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

    args = resolved_args(
        defaults={
            "n_runs": 1,  # runs for the same trainer
            "max_confs": -1,  # crop the number of confs to test
            "max_batchs": -1,  # crop the number of batchs to test
            "conf_idx": None,  # run for this single conf index in the list of configs
            "output_filename": "measured_times.json",  # output file name in data/times/
            "ignore_confs": [],  # list of config indices to ignore
            "overwrite": False,  # overwrite existing file
            "dryrun": False,  # run things but don't load from/save to file
        }
    ).pretty_print()

    configs = deepcopy(ALL_CONFIGS)

    if args.max_confs > 0:
        configs = configs[: args.max_confs]
    elif args.conf_idx is not None:
        if args.max_confs > 0:
            print(
                "\nWARNING: `conf_idx` has precedence over `max_confs`",
                "which will be ignored\n",
            )
        configs = [configs[args.conf_idx]]

    times = {c["overrides"]["note"]: [[] for _ in range(args.n_runs)] for c in configs}

    rewiring_times = {}

    torch.set_grad_enabled(False)
    output_dir = Path(__file__).resolve().parent.parent / "data" / "times"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_json = output_dir / "measured_times.json"

    existing_confs = set()

    if output_json.exists() and not args.overwrite and not args.dryrun:
        with output_json.open("r") as f:
            existing_times = json.load(f)
        times = existing_times["times"]
        rewiring_times = existing_times["rewiring_times"]
        existing_confs = set(k for k in times if any(times[k]))

    for c, config in enumerate(configs):
        note = config["overrides"]["note"]
        if c in existing_confs:
            continue
        trainer = make_trainer(**config, verbose=False)
        trainer.model.eval()
        for r in range(args.n_runs):

            print(
                f"\n\n🔄 Timing {note} ({c+1}/{len(configs)})",
                f"-> Run {r+1}/{args.n_runs}\n\n",
            )

            for b, batch in enumerate(tqdm(trainer.val_loader, desc=note)):
                t = time()
                with torch.cuda.amp.autocast(enabled=trainer.scaler is not None):
                    _ = trainer._forward(batch)
                forward_duration = time() - t
                if trainer.model.module.graph_rewiring:
                    forward_duration -= trainer.model.module.rewiring_time
                    if note not in rewiring_times:
                        rewiring_times[note] = []
                    if len(rewiring_times[note]) == r:
                        rewiring_times[note].append([])
                    rewiring_times[note][r].append(trainer.model.module.rewiring_time)
                times[note][r].append(forward_duration)
                if args.max_batchs > 0 and b > args.max_batchs - 1:
                    break

        if not args.dryrun:
            save_and_print(
                output_json, times, rewiring_times, configs, TRAINER_CONF_OVERRIDES
            )
