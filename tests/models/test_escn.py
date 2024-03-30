import glob
import tempfile
from pathlib import Path, PosixPath

import numpy as np
import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from main import Runner
from ocpmodels.common.utils import build_config
from ocpmodels.models.escn.so3 import CoefficientMapping
from ocpmodels.models.escn.so3 import SO3_Embedding as escn_SO3_Embedding


class DotDict(dict):
    __getattr__ = dict.__getitem__


@pytest.fixture
def escn_args():
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
def escn_config():
    return Path("tests/models/test_configs/test_escn.yml")


class TestTrainESCN:
    def _run_with_args_and_get_events(self, args):
        config = build_config(args, [])
        Runner(args)(config)

        tf_event_files = glob.glob(
            f"{str(args.logdir / 'tensorboard' / '*' / 'events.out*')}"
        )
        assert len(tf_event_files) == 1
        tf_event_file = tf_event_files[0]
        acc = EventAccumulator(tf_event_file)
        acc.Reload()
        return acc

    # train for 300ish steps on a tiny dataset and confirm overfitting
    def test_train_basic(self, escn_args, escn_config):
        with tempfile.TemporaryDirectory() as tempdirname:
            tempdir = Path(tempdirname)
            escn_args["run_dir"] = tempdir
            escn_args["logdir"] = escn_args["run_dir"] / "logs"
            escn_args["config_yml"] = escn_config
            acc = self._run_with_args_and_get_events(escn_args)
            assert acc.Scalars("train/energy_mae")[-1].value < 20

    # train for a few steps and confirm same seeds get same results
    def test_different_seeds(self, escn_args, escn_config):

        with tempfile.TemporaryDirectory() as tempdirname:
            root_tempdir = Path(tempdirname)
            with open(escn_config, "r") as escn_yaml_file:
                yaml_config = yaml.safe_load(escn_yaml_file)

            yaml_config["optim"]["max_epochs"] = 2

            two_epoch_run_yml = root_tempdir / "twoepoch_run.yml"
            with open(str(two_epoch_run_yml), "w") as escn_yaml_file:
                yaml.dump(yaml_config, escn_yaml_file)

            escn_args["config_yml"] = str(two_epoch_run_yml)

            escn_args["seed"] = 0
            escn_args["run_dir"] = root_tempdir / "seed0_take1"
            escn_args["logdir"] = escn_args["run_dir"] / "logs"
            seed0_take1_acc = self._run_with_args_and_get_events(escn_args)

            escn_args["seed"] = 1000
            escn_args["run_dir"] = root_tempdir / "seed1000"
            escn_args["logdir"] = escn_args["run_dir"] / "logs"
            seed1000_acc = self._run_with_args_and_get_events(escn_args)

            escn_args["seed"] = 0
            escn_args["run_dir"] = root_tempdir / "seed0_take2"
            escn_args["logdir"] = escn_args["run_dir"] / "logs"
            seed0_take2_acc = self._run_with_args_and_get_events(escn_args)

            assert not np.isclose(
                seed0_take1_acc.Scalars("train/energy_mae")[-1].value,
                seed1000_acc.Scalars("train/energy_mae")[-1].value,
            )
            assert np.isclose(
                seed0_take1_acc.Scalars("train/energy_mae")[-1].value,
                seed0_take2_acc.Scalars("train/energy_mae")[-1].value,
            )


class TestMPrimaryLPrimary:
    def test_mprimary_lprimary_mappings(self):
        def sign(x):
            return 1 if x >= 0 else -1

        device = torch.device("cpu")
        lmax_list = [6, 8]
        mmax_list = [3, 6]
        for lmax in lmax_list:
            for mmax in mmax_list:
                c = CoefficientMapping([lmax], [mmax], device=device)

                escn_embedding = escn_SO3_Embedding(
                    length=1,
                    lmax_list=[lmax],
                    num_channels=1,
                    device=device,
                    dtype=torch.float32,
                )

                """
                Generate L_primary matrix
                L0: 0.00 ~ L0M0
                L1: -1.01 1.00 1.01 ~ L1M(-1),L1M0,L1M1
                L2: -2.02 -2.01 2.00 2.01 2.02 ~ L2M(-2),L2M(-1),L2M0,L2M1,L2M2
                """
                test_matrix_lp = []
                for l in range(lmax + 1):
                    max_m = min(l, mmax)
                    for m in range(-max_m, max_m + 1):
                        v = l * sign(m) + 0.01 * m  # +/- l . 00 m
                        test_matrix_lp.append(v)

                test_matrix_lp = (
                    torch.tensor(test_matrix_lp)
                    .reshape(1, -1, 1)
                    .to(torch.float32)
                )

                """
                Generate M_primary matrix
                M0: 0.00 , 1.00, 2.00, ... , LMax ~ M0L0, M0L1, .., M0L(LMax)
                M1: 1.01, 2.01, .., LMax.01, -1.01, -2.01, -LMax.01 ~ L1M1, L2M1, .., L(LMax)M1, L1M(-1), L2M(-1), ... , L(LMax)M(-1)
                """
                test_matrix_mp = []
                for m in range(max_m + 1):
                    for l in range(m, lmax + 1):
                        v = l + 0.01 * m  # +/- l . 00 m
                        test_matrix_mp.append(v)
                    if m > 0:
                        for l in range(m, lmax + 1):
                            v = -(l + 0.01 * m)  # +/- l . 00 m
                            test_matrix_mp.append(v)

                test_matrix_mp = (
                    torch.tensor(test_matrix_mp)
                    .reshape(1, -1, 1)
                    .to(torch.float32)
                )

                escn_embedding.embedding = test_matrix_lp.clone()

                escn_embedding._m_primary(c)
                mp = escn_embedding.embedding.clone()
                (test_matrix_mp == mp).all()

                escn_embedding._l_primary(c)
                lp = escn_embedding.embedding.clone()
                (test_matrix_lp == lp).all()
