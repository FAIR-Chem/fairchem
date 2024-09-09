from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from fairchem.core._cli import Runner
from fairchem.core.modules.scaling.fit import compute_scaling_factors
from test_e2e_commons import (
    _run_main,
    oc20_lmdb_train_and_val_from_paths,
    update_yaml_with_dict,
)

from fairchem.core.common.utils import build_config, setup_logging
from fairchem.core.scripts.make_lmdb_sizes import get_lmdb_sizes_parser, make_lmdb_sizes

from fairchem.core.common.flags import flags

setup_logging()


"""
These tests are intended to be as quick as possible and test only that the network is runnable and outputs training+validation to tensorboard output
These should catch errors such as shape mismatches or otherways to code wise break a network
"""


class TestSmoke:
    def smoke_test_train(
        self, input_yaml, tutorial_val_src, world_size, num_workers, otf_norms=False
    ):
        with tempfile.TemporaryDirectory() as tempdirname:
            # first train a very simple model, checkpoint
            train_rundir = Path(tempdirname) / "train"
            train_rundir.mkdir()
            checkpoint_path = str(train_rundir / "checkpoint.pt")
            training_predictions_filename = str(train_rundir / "train_predictions.npz")
            acc = _run_main(
                rundir=str(train_rundir),
                input_yaml=input_yaml,
                update_dict_with={
                    "optim": {
                        "max_epochs": 2,
                        "eval_every": 8,
                        "batch_size": 5,
                        "num_workers": num_workers,
                    },
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                        otf_norms=otf_norms,
                    ),
                },
                save_checkpoint_to=checkpoint_path,
                save_predictions_to=training_predictions_filename,
                world_size=world_size,
            )
            assert "train/energy_mae" in acc.Tags()["scalars"]
            assert "val/energy_mae" in acc.Tags()["scalars"]

            # second load the checkpoint and predict
            predictions_rundir = Path(tempdirname) / "predict"
            predictions_rundir.mkdir()
            predictions_filename = str(predictions_rundir / "predictions.npz")
            _run_main(
                rundir=str(predictions_rundir),
                input_yaml=input_yaml,
                update_dict_with={
                    "optim": {"max_epochs": 2, "eval_every": 8, "batch_size": 5},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                        otf_norms=otf_norms,
                    ),
                },
                update_run_args_with={
                    "mode": "predict",
                    "checkpoint": checkpoint_path,
                },
                save_predictions_to=predictions_filename,
            )

            if otf_norms is True:
                norm_path = glob.glob(
                    str(train_rundir / "checkpoints" / "*" / "normalizers.pt")
                )
                assert len(norm_path) == 1
                assert os.path.isfile(norm_path[0])
                ref_path = glob.glob(
                    str(train_rundir / "checkpoints" / "*" / "element_references.pt")
                )
                assert len(ref_path) == 1
                assert os.path.isfile(ref_path[0])

            # verify predictions from train and predict are identical
            energy_from_train = np.load(training_predictions_filename)["energy"]
            energy_from_checkpoint = np.load(predictions_filename)["energy"]
            npt.assert_allclose(
                energy_from_train, energy_from_checkpoint, rtol=1e-6, atol=1e-6
            )

    def test_gemnet_fit_scaling(self, configs, tutorial_val_src):

        with tempfile.TemporaryDirectory() as tempdirname:
            # (1) generate scaling factors for gemnet config
            config_yaml = f"{tempdirname}/train_and_val_on_val.yml"
            scaling_pt = f"{tempdirname}/scaling.pt"
            # run
            parser = flags.get_parser()
            args, override_args = parser.parse_known_args(
                [
                    "--mode",
                    "train",
                    "--seed",
                    "100",
                    "--config-yml",
                    config_yaml,
                    "--cpu",
                    "--checkpoint",
                    scaling_pt,
                ]
            )
            update_yaml_with_dict(
                configs["gemnet_oc"],
                config_yaml,
                update_dict_with={
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
            )
            config = build_config(args, override_args)

            # (2) if existing scaling factors are present remove them
            if "scale_file" in config["model"]:
                config["model"].pop("scale_file")

            compute_scaling_factors(config)

            # (3) try to run the config with the newly generated scaling factors
            _ = _run_main(
                rundir=tempdirname,
                update_dict_with={
                    "optim": {"max_epochs": 1},
                    "model": {"use_pbc_single": True, "scale_file": scaling_pt},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                input_yaml=configs["gemnet_oc"],
            )

    # not all models are tested with otf normalization estimation
    # only gemnet_oc, escn, equiformer, and their hydra versions
    @pytest.mark.parametrize(
        ("model_name", "otf_norms"),
        [
            ("schnet", False),
            ("scn", False),
            ("gemnet_dt", False),
            ("gemnet_dt_hydra", False),
            ("gemnet_dt_hydra_grad", False),
            ("gemnet_oc", False),
            ("gemnet_oc", True),
            ("gemnet_oc_hydra", False),
            ("gemnet_oc_hydra", True),
            ("gemnet_oc_hydra_grad", False),
            ("gemnet_oc_hydra_energy_only", False),
            ("dimenet++", False),
            ("dimenet++_hydra", False),
            ("painn", False),
            ("painn_hydra", False),
            ("escn", False),
            ("escn", True),
            ("escn_hydra", False),
            ("escn_hydra", True),
            ("equiformer_v2", False),
            ("equiformer_v2", True),
            ("equiformer_v2_hydra", False),
            ("equiformer_v2_hydra", True),
        ],
    )
    def test_train_and_predict(
        self,
        model_name,
        otf_norms,
        configs,
        tutorial_val_src,
    ):
        # test without ddp
        self.smoke_test_train(
            input_yaml=configs[model_name],
            tutorial_val_src=tutorial_val_src,
            otf_norms=otf_norms,
            world_size=0,
            num_workers=2,
        )
        # test with ddp but no wokers
        self.smoke_test_train(
            input_yaml=configs[model_name],
            tutorial_val_src=tutorial_val_src,
            otf_norms=otf_norms,
            world_size=1,
            num_workers=0,
        )

    def test_use_pbc_single(self, configs, tutorial_val_src, torch_deterministic):
        with tempfile.TemporaryDirectory() as tempdirname:
            tempdir = Path(tempdirname)
            extra_args = {"seed": 0}
            _ = _run_main(
                rundir=str(tempdir),
                update_dict_with={
                    "optim": {"max_epochs": 1},
                    "model": {"use_pbc_single": True},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with=extra_args,
                input_yaml=configs["equiformer_v2"],
            )

    def test_max_num_atoms(self, configs, tutorial_val_src, torch_deterministic):
        with tempfile.TemporaryDirectory() as tempdirname:
            tempdir = Path(tempdirname)
            extra_args = {"seed": 0}
            with pytest.raises(AssertionError):
                _ = _run_main(
                    rundir=str(tempdir),
                    update_dict_with={
                        "optim": {"max_epochs": 1},
                        "model": {"backbone": {"max_num_elements": 2}},
                        "dataset": oc20_lmdb_train_and_val_from_paths(
                            train_src=str(tutorial_val_src),
                            val_src=str(tutorial_val_src),
                            test_src=str(tutorial_val_src),
                        ),
                    },
                    update_run_args_with=extra_args,
                    input_yaml=configs["equiformer_v2_hydra"],
                )

    @pytest.mark.parametrize(
        ("world_size", "ddp"),
        [
            pytest.param(
                2,
                True,
            ),
            pytest.param(0, False),
        ],
    )
    def test_ddp(
        self,
        world_size,
        ddp,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        with tempfile.TemporaryDirectory() as tempdirname:
            tempdir = Path(tempdirname)
            extra_args = {"seed": 0}
            if not ddp:
                extra_args["no_ddp"] = True
            _ = _run_main(
                rundir=str(tempdir),
                update_dict_with={
                    "optim": {"max_epochs": 1},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with=extra_args,
                input_yaml=configs["equiformer_v2"],
                world_size=world_size,
            )

    @pytest.mark.parametrize(
        ("world_size", "ddp"),
        [
            pytest.param(2, True),
            pytest.param(0, False),
        ],
    )
    def test_balanced_batch_sampler_ddp(
        self, world_size, ddp, configs, tutorial_val_src, torch_deterministic
    ):
        # make dataset metadata
        parser = get_lmdb_sizes_parser()
        args, override_args = parser.parse_known_args(
            ["--data-path", str(tutorial_val_src)]
        )
        make_lmdb_sizes(args)

        with tempfile.TemporaryDirectory() as tempdirname:
            tempdir = Path(tempdirname)
            extra_args = {"seed": 0}
            if not ddp:
                extra_args["no_ddp"] = True
            _ = _run_main(
                rundir=str(tempdir),
                update_dict_with={
                    "optim": {"max_epochs": 1, "load_balancing": "atoms"},
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                        test_src=str(tutorial_val_src),
                    ),
                },
                update_run_args_with=extra_args,
                input_yaml=configs["equiformer_v2"],
                world_size=world_size,
            )

    # train for a few steps and confirm same seeds get same results
    def test_different_seeds(
        self,
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
        ("model_name", "expected_energy_mae", "expected_force_mae"),
        [
            pytest.param("gemnet_oc", 0.41, 0.06, id="gemnet_oc"),
            pytest.param("escn", 0.41, 0.06, id="escn"),
            pytest.param("equiformer_v2", 0.41, 0.06, id="equiformer_v2"),
        ],
    )
    def test_train_optimization(
        self,
        model_name,
        expected_energy_mae,
        expected_force_mae,
        configs,
        tutorial_val_src,
        torch_deterministic,
    ):
        with tempfile.TemporaryDirectory() as tempdirname:
            acc = _run_main(
                rundir=tempdirname,
                input_yaml=configs[model_name],
                update_dict_with={
                    "dataset": oc20_lmdb_train_and_val_from_paths(
                        train_src=str(tutorial_val_src),
                        val_src=str(tutorial_val_src),
                    ),
                },
            )
            assert acc.Scalars("train/energy_mae")[-1].value < expected_energy_mae
            assert acc.Scalars("train/forces_mae")[-1].value < expected_force_mae
