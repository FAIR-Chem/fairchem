import copy
import os
import time
from pathlib import Path
from shutil import copyfile, move

import yaml
from orion.client import build_experiment
from orion.core.utils.exceptions import ReservationRaceCondition

from ocpmodels.common.utils import ROOT, RUNS_DIR, unflatten_dict


def apply_mult_factor(orion_hparams, mult_factor_dict, sep="."):
    """
    Multiplies all values of orion_hparams listed in mult_factor_dict["targets"]
    by mult_factor_dict["value"].

    eg:
    >>> orion_hparams = {
        "model/hidden_channels": 4,
        "model/num_layers": 4,
        "optim/batch_size": 4,
        "optim/lr_initial": 0.001,
        "frame_averaging": "",
    }

    >>> mult_factor_dict = {"value": 32, "targets": "hidden_channels, batch_size"}

    >>> apply_mult_factor(orion_hparams, mult_factor_dict, sep="/")
    {
        "model/hidden_channels": 128,
        "model/num_layers": 4,
        "optim/batch_size": 128,
        "optim/lr_initial": 0.001,
        "frame_averaging": ""
    }

    Args:
        orion_hparams (_type_): _description_
        mult_factor_dict (_type_): _description_
        sep (str, optional): _description_. Defaults to ".".

    Returns:
        _type_: _description_
    """
    if not mult_factor_dict:
        return orion_hparams
    if not isinstance(mult_factor_dict, dict):
        print(
            f">>> Warning: ignoring apply_mult_factor, not a dict: {mult_factor_dict}."
        )
    if "value" not in mult_factor_dict or "targets" not in mult_factor_dict:
        print(
            ">>> Warning: ignoring apply_mult_factor, "
            + " missing 'value' or 'targets' keys: {}.".format(mult_factor_dict)
        )
    value, targets = mult_factor_dict["value"], mult_factor_dict["targets"]
    targets = set([t.strip() for t in targets.split(",")])
    updated_hparams = copy.deepcopy(orion_hparams)
    for k, v in orion_hparams.items():
        target = k.split(sep)[-1]
        if target in targets:
            updated_hparams[k] = v * value
    return updated_hparams


def set_max_fidelity(hparams, orion_exp):
    for p, prior in orion_exp.space.items():
        if prior.type == "fidelity":
            keys = p.split("/")
            if len(keys) == 1:
                hparams[f"fidelity_{p}"] = prior.high
            elif len(keys) == 2:
                if keys[0] not in hparams:
                    hparams[keys[0]] = {}
                hparams[keys[0]][f"fidelity_{keys[1]}"] = prior.high
            else:
                print("Error: fidelity parameters must be at most 2 levels deep.")
    return hparams


def sample_orion_hparams(orion_exp, trainer_config):
    hparams = {}
    orion_trial = None
    try:
        orion_trial = orion_exp.suggest(1)
        print(
            "\n🚨  Orion reservation race condition detected. Exiting",
            "and deleting run dir",
        )
        hparams = set_max_fidelity(
            unflatten_dict(
                apply_mult_factor(
                    orion_trial.params,
                    trainer_config.get("orion_mult_factor"),
                    sep="/",
                ),
                sep="/",
            ),
            orion_exp,
        )
        hparams["orion_hash_params"] = orion_trial.hash_params
        hparams["orion_unique_exp_name"] = orion_exp.name
    except ReservationRaceCondition:
        hparams["orion_race_condition"] = True
        import wandb

        if wandb.run is not None:
            if wandb.run.tags:
                wandb.run.tags = wandb.run.tags + ("RaceCondition",)
            else:
                wandb.run.tags = ("RaceCondition",)
    return hparams, orion_trial


def get_and_move_orion_db_path(exp_name):
    db_id = "".join([c for c in exp_name if c.isalnum() or c in "_-."])
    db_file = f"{db_id}_db.pkl" if not db_id.endswith("_db.pkl") else db_id
    scratch_db = RUNS_DIR.parent / "orion" / "storage" / db_file
    scratch_db.parent.mkdir(parents=True, exist_ok=True)
    if not scratch_db.exists():
        home_db = ROOT / f"data/orion/storage/{db_file}"

        if not home_db.exists():
            return scratch_db

        lock_file = home_db.parent / f"{db_file}.cp_lock"
        if not lock_file.exists():
            lock_file.touch()
            copyfile(home_db, scratch_db)
            move(home_db, home_db.parent / f"{db_file}.bak")
            os.symlink(str(scratch_db), str(home_db))
            print("Copied and symlinked db from home to scratch.")
            lock_file.unlink()

        while lock_file.exists():
            print("Waiting for lock to be released...")
            time.sleep(1)

    return scratch_db


def load_orion_exp(args):
    exp_config = yaml.safe_load(Path(args.orion_exp_config_path).read_text())

    assert args.orion_unique_exp_name or exp_config.get(
        "unique_exp_name"
    ), "Must provide orion_unique_exp_name in the command-line or the config file."

    print(f"🔎 Orion Experiment Config:\n{yaml.dump(exp_config)}")
    exp_name = args.orion_unique_exp_name or exp_config["unique_exp_name"]
    db_id = "".join([c for c in exp_name if c.isalnum() or c in "_-."])
    db_path = get_and_move_orion_db_path(db_id)
    experiment = build_experiment(
        storage={
            "database": {
                "host": str(db_path),
                "type": "pickleddb",
            }
        },
        name=exp_name,
        space=exp_config["space"],
        algorithms=exp_config["algorithms"],
    )
    return experiment


def continue_orion_exp(trainer_config):
    if not trainer_config.get("orion_exp_config_path"):
        return trainer_config

    if "orion_hash_params" not in trainer_config:
        faulty_path = Path(trainer_config["run_dir"]) / "faulty_trainer_config.yaml"
        print(
            "\n\nWARNING: trainer_config has 'orion_exp_config_path'",
            "but no 'orion_hash_params'.",
            "This can lead to inconsistencies.",
            f"You should investigate the faulty config in:\n{str(faulty_path)}\n\n",
        )
        faulty_path.write_text(yaml.dump(trainer_config))
        return trainer_config

    hash_params = trainer_config["orion_hash_params"]
    exp_name = trainer_config["orion_unique_exp_name"]
    id_file = f"{exp_name}--{hash_params}.unique"
    (Path(trainer_config["run_dir"]) / id_file).touch()
    base_dir = Path(trainer_config["run_dir"]).parent
    existing_id_files = list(base_dir.glob(f"*/{id_file}"))

    latest_dirs = sorted(
        [
            f.parent
            for f in existing_id_files
            if float(f.parent.name) != float(trainer_config["job_id"])
        ],
        key=lambda f: float(f.name),
    )

    if not latest_dirs:
        print("\n😅 No previous Orion trial matched for unique file: ", id_file)
        return trainer_config

    resume_dir = latest_dirs[-1]

    resume_ckpts = sorted(
        [f for f in (resume_dir / "checkpoints").glob("checkpoint-*")],
        key=lambda f: float(f.stem.split("-")[-1]),
    )

    if not resume_ckpts:
        print(f"🥶 Warning: No checkpoint found in {str(resume_dir)}. Not resuming.")
        return trainer_config

    trainer_config["checkpoint"] = str(resume_ckpts[-1])
    resume_url = (resume_dir / "wandb_url.txt").read_text().strip()
    trainer_config["wandb_resume_id"] = resume_url.split("/runs/")[-1]

    print(
        f"\n🎁 Found {len(resume_ckpts)} existing Orion runs.",
        "Resuming from latest:",
        str(resume_dir),
        "\nOn wandb run:",
        resume_url,
    )
    print("Based on unique file id:", id_file)
    print("Continuing from checkpoint:", trainer_config["checkpoint"], end="\n\n")
    return trainer_config
