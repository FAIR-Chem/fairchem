import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

if Path.cwd().name == "scripts":
    sys.path.append("..")

from time import time

from minydra import resolved_args
from tqdm import tqdm

from ocpmodels.common.utils import make_trainer

if __name__ == "__main__":

    args = resolved_args(
        defaults={
            "n_runs": 1,
            "max_confs": -1,
            "max_batchs": -1,
        }
    ).pretty_print()

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
                "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
            ],
            "overrides": {**trainer_conf_overrides, "note": "schnet Baseline"},
        },
        {
            "str_args": [
                "--mode=train",
                "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
            ],
            "overrides": {
                **trainer_conf_overrides,
                "note": "dimenet_plus_plus Baseline",
            },
        },
        {
            "str_args": [
                "--mode=train",
                "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
            ],
            "overrides": {**trainer_conf_overrides, "note": "forcenet Baseline"},
        },
        {
            "str_args": [
                "--mode=train",
                "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
                "--model.graph_rewiring=remove-tag-0",
            ],
            "overrides": {
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
                "note": "forcenet Tag Hidden Channels 64",
            },
        },
        {
            "str_args": [
                "--mode=train",
                "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
                "--model.phys_embeds=true",
            ],
            "overrides": {**trainer_conf_overrides, "note": "schnet Phys Embeds Fixed"},
        },
        {
            "str_args": [
                "--mode=train",
                "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
                "--model.phys_embeds=true",
            ],
            "overrides": {
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
            "overrides": {**trainer_conf_overrides, "note": "schnet"},
        },
        {
            "str_args": [
                "--mode=train",
                "--config-yml=configs/is2re/all/dimenet_plus_plus/new_dpp.yml",
                "--model.phys_embeds=true",
                "--model.phys_hidden_channels=32",
            ],
            "overrides": {**trainer_conf_overrides, "note": "dimenet_plus_plus"},
        },
        {
            "str_args": [
                "--mode=train",
                "--config-yml=configs/is2re/all/forcenet/new_forcenet.yml",
                "--model.phys_embeds=true",
                "--model.phys_hidden_channels=32",
            ],
            "overrides": {
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
            "overrides": {**trainer_conf_overrides, "note": "schnet All embeddings"},
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
                **trainer_conf_overrides,
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
            "overrides": {**trainer_conf_overrides, "note": "forcenet All embeddings"},
        },
        {
            "str_args": [
                "--mode=train",
                "--config-yml=configs/is2re/all/schnet/new_schnet.yml",
                "--model.energy_head=weighted-av-initial-embeds",
            ],
            "overrides": {
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
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
                **trainer_conf_overrides,
                "note": "forcenet Energy Head: Graclus",
            },
        },
    ]
    if args.max_confs > 0:
        configs = configs[: args.max_confs]

    times = {c["overrides"]["note"]: [[] for _ in range(args.n_runs)] for c in configs}

    rewiring_times = {}

    torch.set_grad_enabled(False)

    for c, config in enumerate(configs):
        for r in range(args.n_runs):
            trainer = make_trainer(**config)
            trainer.model.eval()
            note = config["overrides"]["note"]

            print(
                f"\n\n🔄 Timing {note} ({c+1}/{len(configs)}) -> Run {r+1}/{args.n_runs}\n\n"
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
    for note, s in stats.items():
        n = note + "  " + "⸱" * (50 - len(note))
        print(f"• {n} {s['mean']:8.4f}s +/- {s['std']:.4f}s")

    print("\nREWIRINGS")
    for note, s in rewiring_stats.items():
        n = note + "  " + "⸱" * (50 - len(note))
        print(f"• {n} {s['mean']:8.4f}s +/- {s['std']:.4f}s")

    output = Path(__file__).resolve().parent.parent / "data" / "times"
    output.mkdir(exist_ok=True, parents=True)
    now = str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    with open(output / f"{now}_times.json", "w") as f:
        out_dict = {
            "configs": configs,
            "trainer_conf_overrides": trainer_conf_overrides,
            "times": times,
            "val_times": val_times,
            "stats": stats,
            "rewiring_times": rewiring_times,
            "epoch_rewiring_times": epoch_rewiring_times,
            "rewiring_stats": rewiring_stats,
        }
        json.dump(out_dict, f)
