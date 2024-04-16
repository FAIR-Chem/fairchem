import collections.abc
import copy
import glob
import os
import tempfile
from pathlib import Path, PosixPath
from typing import List

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


def oc20_lmdb_train_and_val_from_paths(train_src, val_src, test_src=None):
    datasets = {}
    if train_src is not None:
        datasets["train"] = oc20_lmdb_train_config_from_path(train_src)
    if val_src is not None:
        datasets["val"] = oc20_lmdb_val_config_from_path(val_src)
    if test_src is not None:
        datasets["test"] = oc20_lmdb_val_config_from_path(val_src)
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
    if update_with is not None:
        yaml_config = merge_dictionary(yaml_config, update_with)
    if output_yaml is None:
        output_yaml = input_yaml
    with open(str(output_yaml), "w") as escn_yaml_file:
        yaml.dump(yaml_config, escn_yaml_file)


def _run_main(
    rundir,
    base_command_line_args,
    input_yaml,
    update_dict_with=None,
    update_run_args_with=None,
    save_checkpoint_to=None,
    save_predictions_to=None,
):
    config_yaml = Path(rundir) / f"train_and_val_on_val.yml"

    config_change_values(
        input_yaml=input_yaml,
        output_yaml=config_yaml,
        update_with=update_dict_with,
    )
    run_args = {
        "run_dir": rundir,
        "logdir": f"{rundir}/logs",
        "config_yml": config_yaml,
    }
    if update_run_args_with is not None:
        run_args.update(update_run_args_with)
    run_with_args(
        base_command_line_args, run_args,
    )
    if save_checkpoint_to is not None:
        checkpoints = glob.glob(f"{rundir}/checkpoints/*/checkpoint.pt")
        assert len(checkpoints) == 1
        os.rename(checkpoints[0], save_checkpoint_to)
    if save_predictions_to is not None:
        predictions_filenames = glob.glob(
            f"{rundir}/results/*/s2ef_predictions.npz"
        )
        assert len(predictions_filenames) == 1
        os.rename(predictions_filenames[0], save_predictions_to)
    return get_tensorboard_log_values(f"{rundir}/logs",)


@pytest.fixture(scope="class")
def torch_tempdir(tmpdir_factory):
    return tmpdir_factory.mktemp("torch_tempdir")


"""
These tests are intended to be as quick as possible and test only that the network is runnable and outputs training+validation to tensorboard output
These should catch errors such as shape mismatches or otherways to code wise break a network
"""


class TestSmoke:
    def smoke_test_train(
        self, model_name, base_command_line_args, input_yaml, tutorial_val_src,
    ):
        with tempfile.TemporaryDirectory() as tempdirname:
            # first train a very simple model, checkpoint
            train_rundir = Path(tempdirname) / "train"
            train_rundir.mkdir()
            checkpoint_path = str(train_rundir / "checkpoint.pt")
            training_predictions_filename = str(
                train_rundir / "train_predictions.npz"
            )
            acc = _run_main(
                rundir=str(train_rundir),
                base_command_line_args=base_command_line_args,
                input_yaml=input_yaml,
                update_dict_with={
                    "optim": {"max_epochs": 2, "eval_every": 8},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                save_checkpoint_to=checkpoint_path,
                save_predictions_to=training_predictions_filename,
            )
            assert "train/energy_mae" in acc.Tags()["scalars"]
            assert "val/energy_mae" in acc.Tags()["scalars"]

            # second load the checkpoint and predict
            predictions_rundir = Path(tempdirname) / "predict"
            predictions_rundir.mkdir()
            predictions_filename = str(predictions_rundir / "predictions.npz")
            _run_main(
                rundir=str(predictions_rundir),
                base_command_line_args=base_command_line_args,
                input_yaml=input_yaml,
                update_dict_with={
                    "optim": {"max_epochs": 2, "eval_every": 8},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with={
                    "mode": "predict",
                    "checkpoint": checkpoint_path,
                },
                save_predictions_to=predictions_filename,
            )

            # verify predictions from train and predict are identical
            energy_from_train = np.load(training_predictions_filename)[
                "energy"
            ]
            energy_from_checkpoint = np.load(predictions_filename)["energy"]
            assert np.isclose(energy_from_train, energy_from_checkpoint).all()

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param("gemnet", id="gemnet"),
            pytest.param("escn", id="escn"),
            pytest.param("equiformer_v2", id="equiformer_v2"),
        ],
    )
    def test_train_and_predict(
        self, model_name, base_command_line_args, configs, tutorial_val_src,
    ):
        self.smoke_test_train(
            model_name=model_name,
            base_command_line_args=base_command_line_args,
            input_yaml=configs[model_name],
            tutorial_val_src=tutorial_val_src,
        )

    # train for a few steps and confirm same seeds get same results
    def test_different_seeds(
        self,
        base_command_line_args,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        with tempfile.TemporaryDirectory() as tempdirname:
            tempdir = Path(tempdirname)

            seed0_take1_rundir = tempdir / "seed0take1"
            seed0_take1_rundir.mkdir()
            seed0_take1_acc = _run_main(
                rundir=str(seed0_take1_rundir),
                base_command_line_args=base_command_line_args,
                update_dict_with={
                    "optim": {"max_epochs": 2},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with={"seed": 0},
                input_yaml=configs["escn"],
            )

            seed1000_rundir = tempdir / "seed1000"
            seed1000_rundir.mkdir()
            seed1000_acc = _run_main(
                rundir=str(seed1000_rundir),
                base_command_line_args=base_command_line_args,
                update_dict_with={
                    "optim": {"max_epochs": 2},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with={"seed": 1000},
                input_yaml=configs["escn"],
            )

            seed0_take2_rundir = tempdir / "seed0_take2"
            seed0_take2_rundir.mkdir()
            seed0_take2_acc = _run_main(
                rundir=str(seed0_take2_rundir),
                base_command_line_args=base_command_line_args,
                update_dict_with={
                    "optim": {"max_epochs": 2},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with={"seed": 0},
                input_yaml=configs["escn"],
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
    @pytest.mark.parametrize(
        "model_name,expected_energy_mae,expected_force_mae",
        [
            pytest.param("gemnet", 0.4, 0.06, id="gemnet"),
            pytest.param("escn", 0.4, 0.06, id="escn"),
            pytest.param("equiformer_v2", 0.4, 0.06, id="equiformer_v2"),
        ],
    )
    def test_train_optimization(
        self,
        model_name,
        expected_energy_mae,
        expected_force_mae,
        base_command_line_args,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        with tempfile.TemporaryDirectory() as tempdirname:
            acc = _run_main(
                rundir=tempdirname,
                base_command_line_args=base_command_line_args,
                input_yaml=configs[model_name],
                update_dict_with={
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                    ),
                },
            )
            assert (
                acc.Scalars("train/energy_mae")[-1].value < expected_energy_mae
            )
            assert (
                acc.Scalars("train/forces_mae")[-1].value < expected_force_mae
            )
