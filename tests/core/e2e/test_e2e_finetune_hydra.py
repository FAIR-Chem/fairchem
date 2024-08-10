from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
from test_e2e_commons import _run_main, oc20_lmdb_train_and_val_from_paths

from fairchem.core.models.finetune_hydra import FTHYDRA_NAME, FineTuneMode, FTConfig


@pytest.fixture()
def tutorial_val_src(tutorial_dataset_path):
    return tutorial_dataset_path / "s2ef/val_20"


def make_checkpoint(data_source, seed) -> str:
    # first train a tiny eqv2 model to get a checkpoint
    eqv2_yml = Path("tests/core/models/test_configs/test_equiformerv2_hydra.yml")
    tempdir = tempfile.TemporaryDirectory()
    ck_path = os.path.join(tempdir.name, "checkpoint.pt")
    _run_main(
        tempdir.name,
        eqv2_yml,
        update_dict_with={
            "optim": {
                "max_epochs": 1,
                "eval_every": 8,
                "batch_size": 1,
                "num_workers": 0,
            },
            "dataset": oc20_lmdb_train_and_val_from_paths(
                train_src=str(data_source),
                val_src=str(data_source),
                test_src=str(data_source),
                otf_norms=False,
            ),
        },
        update_run_args_with={"seed": seed},
        save_checkpoint_to=ck_path,
        world_size=0,
    )
    assert os.path.isfile(ck_path)
    return ck_path, tempdir


def run_main_with_ft_hydra(tempdir, yaml, data_src, seed, ft_config, output_checkpoint):
    _run_main(
        tempdir.name,
        yaml,
        update_dict_with={
            "optim": {
                "max_epochs": 1,
                "eval_every": 8,
                "batch_size": 1,
                "num_workers": 0,
                "lr_initial": 0.0 # don't learn anything
            },
            "dataset": oc20_lmdb_train_and_val_from_paths(
                train_src=str(data_src),
                val_src=str(data_src),
                test_src=str(data_src),
                otf_norms=False,
            ),
            "model": {
                "name": FTHYDRA_NAME,
                FTConfig.FT_CONFIG_NAME: ft_config,
            }
        },
        update_run_args_with={"seed": seed}, # choose a different seed than the original model
        save_checkpoint_to=output_checkpoint,
        world_size=0,
    )


def test_finetune_hydra_retain_backbone(tutorial_val_src):
    starting_ckpt, original_tmpdir = make_checkpoint(tutorial_val_src, 0)
    old_state_dict = torch.load(starting_ckpt)["state_dict"]
    # now finetune a the model with the checkpoint from the first job
    tempdir2 = tempfile.TemporaryDirectory()
    ft_yml = Path("tests/core/models/test_configs/test_finetune_hydra.yml")
    ck_ft_path = os.path.join(tempdir2.name, "checkpoint_ft.pt")
    ft_config = {
        "mode": FineTuneMode.RETAIN_BACKBONE_ONLY.name,
        "starting_checkpoint": starting_ckpt,
        "heads": {
            "energy": {
                "module": "equiformer_v2_energy_head"
            },
            "forces": {
                "module": "equiformer_v2_force_head"
            }
        }
    }
    run_main_with_ft_hydra(tempdir2, ft_yml, tutorial_val_src, 1000, ft_config, ck_ft_path)
    assert os.path.isfile(ck_ft_path)
    ft_ckpt = torch.load(ck_ft_path)
    assert "config" in ft_ckpt
    assert ft_ckpt["config"]["model"]["name"] == FTHYDRA_NAME
    # check that the backbone weights are the same, and other weights are not the same
    new_state_dict = ft_ckpt["state_dict"]
    for key in new_state_dict:
        if key.startswith("backbone"):
            # backbone should be identical
            assert torch.allclose(new_state_dict[key], old_state_dict[key])
        elif key.startswith("output_heads") and key.endswith("weight"):
            # heads weight should be different because the seeds are different
            assert not torch.allclose(new_state_dict[key], old_state_dict[key])


def test_finetune_hydra_data_only(tutorial_val_src):
    starting_ckpt, original_tmpdir = make_checkpoint(tutorial_val_src, 0)
    old_state_dict = torch.load(starting_ckpt)["state_dict"]
    # now finetune a the model with the checkpoint from the first job
    tempdir2 = tempfile.TemporaryDirectory()
    ft_yml = Path("tests/core/models/test_configs/test_finetune_hydra.yml")
    ck_ft_path = os.path.join(tempdir2.name, "checkpoint_ft.pt")
    ft_config = {
        "mode": FineTuneMode.DATA_ONLY.name,
        "starting_checkpoint": starting_ckpt,
    }
    run_main_with_ft_hydra(tempdir2, ft_yml, tutorial_val_src, 0, ft_config, ck_ft_path)
    assert os.path.isfile(ck_ft_path)
    ft_ckpt = torch.load(ck_ft_path)
    assert "config" in ft_ckpt
    config_model = ft_ckpt["config"]["model"]
    assert config_model["name"] == FTHYDRA_NAME
    # check that the entire model weights are the same
    new_state_dict = ft_ckpt["state_dict"]
    assert len(new_state_dict) == len(old_state_dict)
    for key in new_state_dict:
        assert torch.allclose(new_state_dict[key], old_state_dict[key])
    # check the new checkpoint contains a hydra model
    assert FTConfig.STARTING_MODEL in config_model[FTConfig.FT_CONFIG_NAME]


def test_finetune_from_finetunehydra(tutorial_val_src):
    starting_ckpt, original_tmpdir = make_checkpoint(tutorial_val_src, 0)
    # now finetune a the model with the checkpoint from the first job
    finetune_run1 = tempfile.TemporaryDirectory()
    ft_yml = Path("tests/core/models/test_configs/test_finetune_hydra.yml")
    ck_ft_path = os.path.join(finetune_run1.name, "checkpoint_ft.pt")
    ft_config_1 = {
        "mode": FineTuneMode.DATA_ONLY.name,
        "starting_checkpoint": starting_ckpt,
    }
    run_main_with_ft_hydra(finetune_run1, ft_yml, tutorial_val_src, 0, ft_config_1, ck_ft_path)
    assert os.path.isfile(ck_ft_path)

    # now that we have a second checkpoint, try finetuning again from this checkpoint
    ########################################################################################
    finetune_run2 = tempfile.TemporaryDirectory()
    ck_ft2_path = os.path.join(finetune_run2.name, "checkpoint_ft.pt")
    ft_config_2 = {
        "mode": FineTuneMode.DATA_ONLY.name,
        "starting_checkpoint": ck_ft_path,
    }
    run_main_with_ft_hydra(finetune_run2, ft_yml, tutorial_val_src, 0, ft_config_2, ck_ft2_path)
    ft_ckpt2 = torch.load(ck_ft2_path)
    assert "config" in ft_ckpt2
    config_model = ft_ckpt2["config"]["model"]
    assert config_model["name"] == FTHYDRA_NAME
    old_state_dict = torch.load(ck_ft_path)["state_dict"]
    new_state_dict = ft_ckpt2["state_dict"]
    # the state dicts should still be identical because we made the LR = 0.0
    for key in new_state_dict:
        assert torch.allclose(new_state_dict[key], old_state_dict[key])
