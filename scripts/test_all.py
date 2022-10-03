import os
import shutil
import sys
from pathlib import Path

from minydra import resolved_args

try:
    import ipdb  # noqa: F401

    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
except:  # noqa: E722
    print(
        "`ipdb` is not installed. ",
        "Consider `pip install ipdb` to improve your debugging experience.",
    )

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.common.utils import make_script_trainer

COLS = shutil.get_terminal_size().columns


def clean_previous_line():
    print("\033[F" + " " * COLS, end="\r")


if __name__ == "__main__":
    args = resolved_args(
        defaults={
            "skip_features": -1,  # (int) how many features to skip
            "skip_models": -1,  # (int) how many models to skip
            "skip_configs": -1,  # (int) how many final configs to skip
            "ignore_str": "",  # (str) ignore configs containing this string
            "only_str": "",  # (str) only selects configs containing this string
        }
    )

    overrides = {
        "silent": True,
        "logger": "dummy",
        "optim": {"max_epochs": 1, "batch_size": 2},
    }
    models = [
        ["--config-yml=configs/is2re/10k/schnet/new_schnet.yml"],
        ["--config-yml=configs/is2re/10k/dimenet_plus_plus/new_dpp.yml"],
        ["--config-yml=configs/is2re/10k/forcenet/new_forcenet.yml"],
        ["--config-yml=configs/is2re/10k/fanet/fanet.yml"],
        ["--config-yml=configs/is2re/10k/sfarinet/sfarinet.yml"],
    ]

    features = [
        "",
        "--frame_averaging=2d --fa_frames=random",
        "--frame_averaging=2d --fa_frames=det",
        "--frame_averaging=2d --fa_frames=all",
        "--frame_averaging=2d --fa_frames=e3",
        "--frame_averaging=3d",
        "--frame_averaging=da",
        "--graph_rewiring=remove-tag-0",
        "--frame_averaging=3d --graph_rewiring=remove-tag-0",
    ]

    if args.skip_models > 0:
        models = models[args.skip_models :]
    if args.skip_features > 0:
        features = features[args.skip_features :]

    configs = [m + f.split() for m in models for f in features]

    if args.ignore_str:
        if isinstance(args.ignore_str, str):
            args.ignore_str = [args.ignore_str]
        configs = [
            c for c in configs if all(igs not in " ".join(c) for igs in args.ignore_str)
        ]
    if args.only_str:
        configs = [c for c in configs if args.only_str in " ".join(c)]

    if args.skip_configs > 0:
        configs = configs[args.skip_configs :]

    conf_strs = [
        " ".join(conf).replace("--", "").replace("configs/is2re/10k/", "")
        for conf in configs
    ]

    print("🥁 Configs to test:")
    for c, conf in enumerate(configs):
        print(f"  • {c+1} " + conf_strs[c])
        if c and c % len(features) == 0:
            print()

    print()

    nk = len(str(len(configs)))
    for c, conf in enumerate(configs):
        print(f"🔄 Testing config {c+1}/{len(configs)} -> {conf_strs[c]}")
        trainer = make_script_trainer(str_args=conf, overrides=overrides, silent=True)
        is_nan = trainer.train(debug_batches=2)
        clean_previous_line()
        symbol = "✅" if not is_nan else "❌"
        print(f"{symbol} Config {c+1:{nk}}/{len(configs)} -> {conf_strs[c]}")
        print("-" * 10)
