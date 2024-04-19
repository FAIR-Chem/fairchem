import os
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
from minydra import resolved_args

args = resolved_args(
    defaults={
        "skip_features": -1,  # (int) how many features to skip
        "skip_models": -1,  # (int) how many models to skip
        "skip_configs": -1,  # (int) how many final configs to skip
        "ignore_str": "",  # (str) ignore configs containing this string
        "only_str": "",  # (str) only selects configs containing this string
        "traceback": False,  # (bool) print traceback on error
        "n": -1,  # (int) how many configs to run
        "breakpoint": False,  # (bool) call breakpoints on errors
        "help": False,  # (bool) print help
    }
)

if args.help:
    print("Usage: python test_all.py options=value true_flag -false_flag")
    print(
        """
        skip_features -> -1,      # (int) how many features to skip
        skip_models   -> -1,      # (int) how many models to skip
        skip_configs  -> -1,      # (int) how many final configs to skip
        ignore_str    -> "",      # (str) ignore configs containing this string
        only_str      -> "",      # (str) only selects configs containing this string
        traceback     -> False,   # (bool) print traceback on error
        n             -> -1,      # (int) how many configs to run
        breakpoint    -> False,   # (bool) call breakpoints on errors
        help          -> False,   # (bool) print help
        """
    )
    sys.exit(0)

try:
    import ipdb  # noqa: F401

    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
except:  # noqa: E722
    print(
        "`ipdb` is not installed. ",
        "Consider `pip install ipdb` to improve your debugging experience.",
    )

os.environ["ocp_test_env"] = "1"

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.common.utils import get_commit_hash, make_script_trainer
from ocpmodels.trainers.single_trainer import SingleTrainer

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


def isin(key, args):
    return any([key in arg for arg in args])


if __name__ == "__main__":
    command = "python " + " ".join(sys.argv)

    overrides = {
        "silent": True,
        "logger": "dummy",
        "optim": {
            "max_epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "eval_batch_size": 2,
        },
    }
    models = [
        ["--config=dpp-is2re-10k"],
        ["--config=faenet-is2re-10k"],
        ["--config=faenet-qm9-10k"],
        ["--config=faenet-s2ef-2M"],
        ["--config=forcenet-is2re-10k"],
        ["--config=forcenet-s2ef-2M"],
        ["--config=schnet-is2re-10k"],
        ["--config=schnet-qm9-10k"],
        ["--config=sfarinet-is2re-10k"],
        ["--config=sfarinet-qm9-10k"],
        ["--config=sfarinet-s2ef-2M"],
        ["--config=sfarinet-qm7x-1k"],
        ["--config=schnet-qm7x-1k"],
    ]

    features = [
        "",
        "--frame_averaging=2D --fa_method=random",
        "--frame_averaging=2D --fa_method=det --test_ri=True",
        "--frame_averaging=2D --fa_method=all",
        "--frame_averaging=2D --fa_method=se3-all --test_ri=True",
        "--frame_averaging=2D --fa_method=se3-random",
        "--frame_averaging=3D --fa_method=random",
        "--frame_averaging=3D --fa_method=all --test_ri=True",
        "--graph_rewiring=remove-tag-0 --frame_averaging=DA",
        "--frame_averaging=2D --fa_method=random --graph_rewiring=remove-tag-0",
    ]

    singles = [
        "--config=schnet-s2ef-2M --regress_forces=from_energy",
        "--config=dpp-s2ef-2M --regress_forces=from_energy",
        "--config=forcenet-s2ef-2M --regress_forces=from_energy",
        "--config=sfarinet-s2ef-2M --regress_forces=from_energy",
        "--config=faenet-s2ef-2M --regress_forces=from_energy",
        "--config=forcenet-s2ef-2M --regress_forces=direct",
        "--config=sfarinet-s2ef-2M --regress_forces=direct",
        "--config=faenet-s2ef-2M --regress_forces=direct",
        "--config=forcenet-s2ef-2M --regress_forces=direct_with_gradient_target",
        "--config=sfarinet-s2ef-2M --regress_forces=direct_with_gradient_target",
        "--config=faenet-s2ef-2M --regress_forces=direct_with_gradient_target",
        "--config=sfarinet-qm7x-1k --regress_forces=direct",
        "--config=sfarinet-qm7x-1k --regress_forces=direct_with_gradient_target",
        "--config=sfarinet-qm7x-1k --regress_forces=from_energy",
        "--config=faenet-is2re-10k --model.mp_type=base",
        "--config=faenet-is2re-10k --model.mp_type=simple",
        "--config=faenet-is2re-10k --model.mp_type=updownscale",
        # "--config=faenet-is2re-10k --model.edge_embed_type=all_rij --model.mp_type=local_env",
        # "--config=faenet-is2re-10k --model.mp_type=att",
        # "--config=faenet-is2re-10k --model.mp_type=base_with_att",
    ]
    singles = [s.split() for s in singles]

    if args.skip_models > 0:
        models = models[args.skip_models :]
    if args.skip_features > 0:
        features = features[args.skip_features :]

    configs = [m + f.split() for m in models for f in features] + singles

    if args.ignore_str:
        if isinstance(args.ignore_str, str):
            args.ignore_str = [args.ignore_str]
        configs = [
            c for c in configs if all(igs not in " ".join(c) for igs in args.ignore_str)
        ]
    if args.only_str:
        configs = [c for c in configs if re.findall(args.only_str, " ".join(c))]

    configs = [
        c
        for c in configs
        if not (isin("qm7x", c) and isin("graph_rewiring", c))
        and not (isin("qm9", c) and isin("graph_rewiring", c))
    ]

    if args.skip_configs > 0:
        configs = configs[args.skip_configs :]

    if args.n > 0:
        configs = configs[: args.n]

    conf_strs = [
        " ".join(conf).replace("--", "").replace("configs/is2re/10k/", "")
        for conf in configs
    ]
    order = np.argsort(conf_strs)
    configs = [configs[o] for o in order]
    conf_strs = [conf_strs[o] for o in order]

    if not configs:
        print("No configs to run 🥶")
    else:
        print("🥁 Configs to test:")
        current = None
        for c, conf in enumerate(configs):
            model = conf[0].split("=")[1].split("-")[0]
            if current is None:
                current = model
            if model != current:
                print()
                current = model
            print(f"  • {c+1:3} " + conf_strs[c])
        print()

    nk = len(str(len(configs)))
    test_start = time()
    successes = c = 0
    for c, conf in enumerate(configs):
        times = Times()
        conf_start = time()
        print(f"🔄 Testing config {c+1}/{len(configs)} ⇢ {conf_strs[c]}")

        try:
            with times.next("👶 Make", True):
                trainer: SingleTrainer = make_script_trainer(
                    str_args=conf,
                    overrides=overrides,
                    silent=True,
                )

            with times.next("💪 Train"):
                is_nan = trainer.train(debug_batches=2)

            with times.next("🧐 Eval"):
                trainer.eval_all_splits(final=False, debug_batches=2)

            if trainer.test_ri:
                with times.next(" 🎗  Test invariance"):
                    trainer.test_model_symmetries(debug_batches=2)

            clean_previous_line()
            symbol = "✅" if not is_nan else "❌"
            if not is_nan:
                successes += 1

        except KeyboardInterrupt:
            print("\n\nInterrupting. 👋 Bye!")
            break

        except Exception as e:
            print(f"\n{e}\n")
            if args.traceback:
                traceback.print_exc()
            symbol = "❌"
            if args.breakpoint:
                breakpoint()

        os.environ["ocp_test_env"] = ""
        conf_duration = time() - conf_start
        print(
            f"{symbol} {c+1:{nk}}/{len(configs)} {format_times(times)}"
            + f" ⌛️ Total: {format_timer(conf_duration)} ➡︎ {conf_strs[c]}"
        )
        clean_current_line()
        print("-" * 10)

    test_duration = time() - test_start
    emo = "🎉" if successes == len(configs) else "😢"
    print(
        f"\n\n{emo} `{command}` finished testing {c+1 if c else 0}/{len(configs)}"
        + f" configs in {format_timer(test_duration)}"
        + f" on commit {get_commit_hash()}. {successes} succeeded."
        + f" [{str(datetime.now()).split('.')[0]}]"
    )
