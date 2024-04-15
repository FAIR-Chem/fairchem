import copy
import glob
import os
import tempfile
from pathlib import Path, PosixPath
from typing import List

import collections.abc

import numpy as np
import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from main import Runner
from ocpmodels.common.utils import build_config, setup_logging
from ocpmodels.datasets import TrajectoryLmdbDataset
from ocpmodels.models.escn.so3 import CoefficientMapping
from ocpmodels.models.escn.so3 import SO3_Embedding as escn_SO3_Embedding
from ocpmodels.trainers import OCPTrainer

from ocpmodels import models  # isort: skip
from ocpmodels.common import logger  # isort: skip


setup_logging()


class DotDict(dict):
    __getattr__ = dict.__getitem__


@pytest.fixture
def base_command_line_args():
    return DotDict(
        {
            "mode": "train",
            "identifier": "",
            "debug": False,
            "run_dir": "./",
            "print_every": 10,
            "seed": 100,
            "amp": False,
            "checkpoint": None,
            "timestamp_id": None,
            "sweep_yml": None,
            "distributed": False,
            "submit": False,
            "summit": False,
            "logdir": PosixPath("logs"),
            "slurm_partition": "ocp",
            "slurm_mem": 80,
            "slurm_timeout": 72,
            "local_rank": 0,
            "distributed_port": 0,
            "world_size": 1,
            "noddp": False,
            "gp_gpus": 0,
            "cpu": True,
            "num_nodes": 0,
            "num_gpus": 0,
            "distributed_backend": "nccl",
            "no_ddp": False,
        }
    )


@pytest.fixture
def configs():
    return {
        "escn": Path("tests/models/test_configs/test_escn.yml"),
        "gemnet": Path("tests/models/test_configs/test_gemnet.yml"),
        "equiformer_v2": Path(
            "tests/models/test_configs/test_equiformerv2.yml"
        ),
    }


@pytest.fixture
def tutorial_train_src(tutorial_dataset_path):
    return tutorial_dataset_path / "s2ef/train_100"


@pytest.fixture
def tutorial_val_src(tutorial_dataset_path):
    return tutorial_dataset_path / "s2ef/val_20"


def oc20_lmdb_train_config_from_path(src):
    return {
        "src": src,
        "normalize_labels": True,
        "target_mean": -0.7554450631141663,
        "target_std": 2.887317180633545,
        "grad_target_mean": 0.0,
        "grad_target_std": 2.887317180633545,
    }


def oc20_lmdb_val_config_from_path(src):
    return {"src": src}


def oc20_lmdb_train_and_val_from_paths(train_src, val_src):
    datasets = {}
    if train_src is not None:
        datasets["train"] = oc20_lmdb_train_config_from_path(train_src)
    if val_src is not None:
        datasets["val"] = oc20_lmdb_val_config_from_path(val_src)
    return datasets


def run_with_args(args, update_args):
    args = DotDict(args.copy())
    args.update(update_args)
    config = build_config(args, [])
    Runner(args)(config)


def get_tensorboard_log_files(logdir):
    return glob.glob(f"{logdir}/tensorboard/*/events.out*")


def get_tensorboard_log_values(logdir):
    tf_event_files = get_tensorboard_log_files(logdir)
    assert len(tf_event_files) == 1
    tf_event_file = tf_event_files[0]
    acc = EventAccumulator(tf_event_file)
    acc.Reload()
    return acc


def merge_dictionary(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_dictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_change_values(input_yaml, update_with, output_yaml=None):
    with open(input_yaml, "r") as escn_yaml_file:
        yaml_config = yaml.safe_load(escn_yaml_file)
    yaml_config = merge_dictionary(yaml_config, update_with)
    if output_yaml is None:
        output_yaml = input_yaml
    with open(str(output_yaml), "w") as escn_yaml_file:
        yaml.dump(yaml_config, escn_yaml_file)


def update_config_to_train_and_val_on_val(
    input_yaml, output_yaml, tutorial_val_src
):
    config_change_values(
        input_yaml=str(input_yaml),
        update_with={
            "dataset": oc20_lmdb_train_and_val_from_paths(
                train_src=str(tutorial_val_src),
                val_src=str(tutorial_val_src),
            )
        },
        output_yaml=output_yaml,
    )


def _run_train_and_val_on_val(
    base_command_line_args,
    configs,
    model_name,
    tutorial_val_src,
    update_dict_with={},
    update_run_args_with={},
):
    with tempfile.TemporaryDirectory() as tempdirname:
        yml = Path(tempdirname) / f"{model_name}_train_and_val_on_val.yml"
        update_config_to_train_and_val_on_val(
            input_yaml=configs[model_name],
            output_yaml=yml,
            tutorial_val_src=tutorial_val_src,
        )
        if len(update_dict_with) > 0:
            config_change_values(input_yaml=yml, update_with=update_dict_with)
        run_args = {
            "run_dir": tempdirname,
            "logdir": f"{tempdirname}/logs",
            "config_yml": yml,
        }
        run_args.update(update_run_args_with)
        run_with_args(
            base_command_line_args,
            run_args,
        )
        return get_tensorboard_log_values(
            f"{tempdirname}/logs",
        )


"""
These tests are intended to be as quick as possible and test only that the network is runnable and outputs training+validation to tensorboard output
These should catch errors such as shape mismatches or otherways to code wise break a network
"""


class TestSmoke:
    def test_gemnet(
        self,
        base_command_line_args,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        acc = _run_train_and_val_on_val(
            base_command_line_args,
            configs,
            model_name="gemnet",
            tutorial_val_src=tutorial_val_src,
            update_dict_with={"optim": {"max_epochs": 2, "eval_every": 3}},
        )
        assert "train/energy_mae" in acc.Tags()["scalars"]
        assert "val/energy_mae" in acc.Tags()["scalars"]

    def test_escn(
        self,
        base_command_line_args,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        acc = _run_train_and_val_on_val(
            base_command_line_args,
            configs,
            model_name="escn",
            tutorial_val_src=tutorial_val_src,
            update_dict_with={"optim": {"max_epochs": 2, "eval_every": 3}},
        )
        assert "train/energy_mae" in acc.Tags()["scalars"]
        assert "val/energy_mae" in acc.Tags()["scalars"]

    def test_equiformerv2(
        self,
        base_command_line_args,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        acc = _run_train_and_val_on_val(
            base_command_line_args,
            configs,
            model_name="equiformer_v2",
            tutorial_val_src=tutorial_val_src,
            update_dict_with={"optim": {"max_epochs": 2, "eval_every": 3}},
        )
        assert "train/energy_mae" in acc.Tags()["scalars"]
        assert "val/energy_mae" in acc.Tags()["scalars"]

    # train for a few steps and confirm same seeds get same results
    def test_different_seeds(
        self,
        base_command_line_args,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        seed0_take1_acc = _run_train_and_val_on_val(
            base_command_line_args,
            configs,
            model_name="escn",
            tutorial_val_src=tutorial_val_src,
            update_dict_with={"optim": {"max_epochs": 2}},
            update_run_args_with={"seed": 0},
        )

        seed1000_acc = _run_train_and_val_on_val(
            base_command_line_args,
            configs,
            model_name="escn",
            tutorial_val_src=tutorial_val_src,
            update_dict_with={"optim": {"max_epochs": 2}},
            update_run_args_with={"seed": 1000},
        )

        seed0_take2_acc = _run_train_and_val_on_val(
            base_command_line_args,
            configs,
            model_name="escn",
            tutorial_val_src=tutorial_val_src,
            update_dict_with={"optim": {"max_epochs": 2}},
            update_run_args_with={"seed": 0},
        )

        assert not np.isclose(
            seed0_take1_acc.Scalars("train/energy_mae")[-1].value,
            seed1000_acc.Scalars("train/energy_mae")[-1].value,
        )
        assert np.isclose(
            seed0_take1_acc.Scalars("train/energy_mae")[-1].value,
            seed0_take2_acc.Scalars("train/energy_mae")[-1].value,
        )


"""
These tests intend to test if optimization is not obviously broken on a time scale of a few minutes
"""


class TestSmallDatasetOptim:

    def test_gemnet(
        self,
        base_command_line_args,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        acc = _run_train_and_val_on_val(
            base_command_line_args,
            configs,
            model_name="gemnet",
            tutorial_val_src=tutorial_val_src,
        )
        assert acc.Scalars("train/energy_mae")[-1].value < 0.4
        assert acc.Scalars("train/forces_mae")[-1].value < 0.06

    def test_escn(
        self,
        base_command_line_args,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        acc = _run_train_and_val_on_val(
            base_command_line_args,
            configs,
            model_name="escn",
            tutorial_val_src=tutorial_val_src,
        )
        assert acc.Scalars("train/energy_mae")[-1].value < 0.4
        assert acc.Scalars("train/forces_mae")[-1].value < 0.06

    def test_equiformerv2(
        self,
        base_command_line_args,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        acc = _run_train_and_val_on_val(
            base_command_line_args,
            configs,
            model_name="equiformer_v2",
            tutorial_val_src=tutorial_val_src,
        )
        assert acc.Scalars("train/energy_mae")[-1].value < 0.4
        assert acc.Scalars("train/forces_mae")[-1].value < 0.06
