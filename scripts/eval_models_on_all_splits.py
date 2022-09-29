import re
import sys
from pathlib import Path

import torch
from minydra import resolved_args

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.common.utils import make_script_trainer, resolve
from ocpmodels.trainers import EnergyTrainer


def read_str_args(run_dir):
    sbatch = list(run_dir.glob("sbatch*.sh"))[0]
    with sbatch.open() as f:
        lines = f.readlines()
    py_args = [line for line in lines if "py_args=" in line][0]
    str_args = py_args.split("py_args=")[1].strip().split(" --")
    str_args = ["--" + a if not a.startswith("--") else a for a in str_args]
    # turn `--mode train` into `--mode=train`
    return [re.sub(r"(--[a-z_\.]+)\s(.+)", r"\1=\2", a) for a in str_args]


def parse_conf(str_args):
    conf = {}
    for a in str_args:
        k = a.split("=")[0].strip().replace("--", "")
        v = a.split("=")[1].strip()
        conf[k] = v
    conf["model"] = conf["config-yml"].split("/")[-1].split(".")[0]
    return conf


TRAINER_CONF_OVERRIDES = {
    "optim": {
        "num_workers": 6,
        "eval_batch_size": 64,
    },
    "logger": "dummy",
}

if __name__ == "__main__":

    args = resolved_args(
        defaults={
            "base_dir": "$SCRATCH/ocp/runs",
            "run_ids": [],
        }
    ).pretty_print()

    base_dir = resolve(args.base_dir)
    run_dirs = [base_dir / str(run_id) for run_id in args.run_ids]
    ckpt_dirs = [run_dir / "checkpoints" for run_dir in run_dirs]
    best_ckpts = [ckpt_dir / "best_checkpoint.pt" for ckpt_dir in ckpt_dirs]
    for bc in best_ckpts:
        assert bc.exists(), f"Best checkpoint {bc} does not exist."

    torch.set_grad_enabled(False)

    for run_dir, ckpt_dir in zip(run_dirs, ckpt_dirs):
        str_args = read_str_args(run_dir)
        overrides = {
            **TRAINER_CONF_OVERRIDES,
        }
        trainer: EnergyTrainer = make_script_trainer(str_args, overrides, verbose=False)
        trainer.model.eval()
        trainer.config["checkpoint_dir"] = str(ckpt_dir)
        print(parse_conf(str_args))
        trainer.eval_all_val_splits(final=True, disable_tqdm=False)

        print("-" * 80 + "\n\n")
