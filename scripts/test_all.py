import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import time

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

from ocpmodels.common.utils import get_commit_hash, make_script_trainer
from ocpmodels.trainers.energy_trainer import EnergyTrainer

COLS = shutil.get_terminal_size().columns


def clean_previous_line(n=1):
    for _ in range(n):
        print("\033[F" + " " * COLS, end="\r")


def clean_current_line():
    print(" " * COLS, end="\r")


def format_timer(t):
    if not isinstance(t, (int, float)):
        t = t.duration

    if t < 60:
        return f"{t:5.2f}s"
    elif t < 3600:
        return f"{t / 60:5.2f}m"
    else:
        return f"{t / 3600:5.2f}h"


def format_times(times):
    if not isinstance(times, dict):
        t = times.times
    else:
        t = times
    s = ""
    for k, v in t.items():
        s += f"{k}: {format_timer(v)} "
    return s


class Timer:
    def __init__(self, name, store={}, is_first=False):
        self.times = store
        self.name = name
        self.is_first = is_first

    def __enter__(self):
        print(f"{' | ' if not self.is_first else ''}{self.name}", end="", flush=True)
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.duration = self.end - self.start
        self.times[self.name] = self.duration
        print(f" {format_timer(self)}", end="", flush=True)


class Times:
    def __init__(self):
        self.times = {}

    def next(self, name, is_first=False):
        return Timer(name, self.times, is_first)


if __name__ == "__main__":
    args = resolved_args(
        defaults={
            "skip_features": -1,  # (int) how many features to skip
            "skip_models": -1,  # (int) how many models to skip
            "skip_configs": -1,  # (int) how many final configs to skip
            "ignore_str": "",  # (str) ignore configs containing this string
            "only_str": "",  # (str) only selects configs containing this string
            "traceback": False,  # (bool) print traceback on error
            "n": -1,  # (int) how many configs to run
        }
    )

    command = "python " + " ".join(sys.argv)

    overrides = {
        "silent": True,
        "logger": "dummy",
        "optim": {"max_epochs": 1, "batch_size": 2},
    }
    models = [
        ["--config=schnet-is2re-10k"],
        ["--config=dpp-is2re-10k"],
        ["--config=forcenet-is2re-10k"],
        ["--config=fanet-is2re-10k"],
        ["--config=sfarinet-is2re-10k"],
        ["--config=sfarinet-s2ef-200k"],
        ["--config=forcenet-s2ef-200k"],
    ]

    features = [
        "",
        "--frame_averaging=2D --fa_frames=random",
        "--frame_averaging=2D --fa_frames=det --test_ri=True",
        "--frame_averaging=2D --fa_frames=all",
        "--frame_averaging=2D --fa_frames=se3-all",
        "--frame_averaging=2D --fa_frames=se3-random",
        "--frame_averaging=3D --fa_frames=random",
        "--frame_averaging=3D --fa_frames=all --test_ri=True",
        "--frame_averaging=DA --test_ri=True",
        "--graph_rewiring=remove-tag-0",
        "--graph_rewiring=remove-tag-0 --frame_averaging=DA",
        "--frame_averaging=2D --fa_frames=random --graph_rewiring=remove-tag-0",
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

    if args.n > 0:
        configs = configs[: args.n]

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
    test_start = time()
    successes = 0
    for c, conf in enumerate(configs):
        times = Times()
        conf_start = time()
        print(f"🔄 Testing config {c+1}/{len(configs)} ⇢ {conf_strs[c]}")

        try:
            with times.next("👶 Make", True):
                trainer: EnergyTrainer = make_script_trainer(
                    str_args=conf,
                    overrides=overrides,
                    silent=True,
                )

            with times.next("💪 Train"):
                is_nan = trainer.train(debug_batches=2)

            with times.next("🧐 Eval"):
                trainer.eval_all_val_splits(final=False, debug_batches=2)

            if trainer.test_ri:
                with times.next(" 🎗  Test invariance"):
                    trainer.test_model_invariance(debug_batches=2)

            clean_previous_line()
            symbol = "✅" if not is_nan else "❌"
            if not is_nan:
                successes += 1

        except Exception as e:
            print(f"\n{e}\n")
            if args.traceback:
                traceback.print_exc()
            symbol = "❌"

        conf_duration = time() - conf_start
        print(
            f"{symbol} {c+1:{nk}}/{len(configs)} {format_times(times)}"
            + f" ⌛️ Total: {format_timer(conf_duration)} ➡︎ {conf_strs[c]}"
        )
        clean_current_line()
        print("-" * 10)

    test_duration = time() - test_start
    print(
        f"\n\n🎉 `{command}` finished testing {len(configs)}"
        + f" configs in {format_timer(test_duration)}"
        + f" on commit {get_commit_hash()}. {successes}/{len(configs)} succeeded."
        + f" [{str(datetime.now()).split('.')[0]}]"
    )
