from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_e2e_commons import _run_main


# TODO add GemNet!
@pytest.mark.parametrize(
    ("model_name", "ddp"),
    [
        ("equiformer_v2_hydra", False),
        ("escn_hydra", False),
        ("equiformer_v2_hydra", True),
        ("escn_hydra", True),
    ],
)
def test_smoke_s2efs_predict(
    model_name, ddp, configs, dummy_binary_dataset_path, tmpdir
):
    # train an s2ef model just to have one
    input_yaml = configs[model_name]
    train_rundir = tmpdir / "train"
    train_rundir.mkdir()
    checkpoint_path = str(train_rundir / "checkpoint.pt")
    training_predictions_filename = str(train_rundir / "train_predictions.npz")

    updates = {
        "task": {"strict_load": False},
        "model": {
            "backbone": {"max_num_elements": 118 + 1},
            "heads": {
                "stress": {
                    "module": "rank2_symmetric_head",
                    "output_name": "stress",
                    "use_source_target_embedding": True,
                }
            },
        },
        "loss_functions": [
            {"energy": {"fn": "mae", "coefficient": 2}},
            {"forces": {"fn": "l2mae", "coefficient": 100}},
            {"stress": {"fn": "mae", "coefficient": 100}},
        ],
        "outputs": {
            "stress": {"level": "system", "irrep_dim": 2, "property": "stress"}
        },
        "evaluation_metrics": {"metrics": {"stress": ["mae", "mae_density"]}},
        "dataset": {
            "train": {
                "src": str(dummy_binary_dataset_path),
                "format": "ase_db",
                "a2g_args": {"r_data_keys": ["energy", "forces", "stress"]},
                "sample_n": 20,
            },
            "val": {
                "src": str(dummy_binary_dataset_path),
                "format": "ase_db",
                "a2g_args": {"r_data_keys": ["energy", "forces", "stress"]},
                "sample_n": 5,
            },
            "test": {
                "src": str(dummy_binary_dataset_path),
                "format": "ase_db",
                "a2g_args": {"r_data_keys": ["energy", "forces", "stress"]},
                "sample_n": 5,
            },
        },
    }

    acc = _run_main(
        rundir=str(train_rundir),
        input_yaml=input_yaml,
        update_dict_with={
            "optim": {
                "max_epochs": 2,
                "eval_every": 4,
                "batch_size": 5,
                "num_workers": 0 if ddp else 2,
            },
            **updates,
        },
        save_checkpoint_to=checkpoint_path,
        save_predictions_to=training_predictions_filename,
        world_size=1 if ddp else 0,
    )
    assert "train/energy_mae" in acc.Tags()["scalars"]
    assert "val/energy_mae" in acc.Tags()["scalars"]

    # now load a checkpoint with an added stress head
    # second load the checkpoint and predict
    predictions_rundir = Path(tmpdir) / "predict"
    predictions_rundir.mkdir()
    predictions_filename = str(predictions_rundir / "predictions.npz")
    _run_main(
        rundir=str(predictions_rundir),
        input_yaml=input_yaml,
        update_dict_with={
            "task": {"strict_load": False},
            "optim": {"max_epochs": 2, "eval_every": 8, "batch_size": 5},
            **updates,
        },
        update_run_args_with={
            "mode": "predict",
            "checkpoint": checkpoint_path,
        },
        save_predictions_to=predictions_filename,
    )
    predictions = np.load(training_predictions_filename)

    for output in updates["outputs"]:
        assert output in predictions

    assert predictions["energy"].shape == (5, 1)
    assert predictions["forces"].shape == (10, 3)
    assert predictions["stress"].shape == (5, 9)
