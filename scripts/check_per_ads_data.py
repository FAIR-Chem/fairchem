"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ocpmodels.common.utils import make_script_trainer
from ocpmodels.trainers import SingleTrainer

if __name__ == "__main__":
    config = {}

    # Customize args
    config["graph_rewiring"] = "remove-tag-0"
    config["frame_averaging"] = "2D"
    config["fa_method"] = "random"  # "random"
    config["test_ri"] = False
    config["optim"] = {"max_epochs": 1}
    config["model"] = {"use_pbc": True}
    checkpoint_path = None

    str_args = sys.argv[1:]
    if all("config" not in arg for arg in str_args):
        str_args.append("--is_debug")
        str_args.append("--config=faenet-is2re-10k")
        str_args.append("--adsorbates={'*O', '*OH', '*OH2', '*H'}")
        # str_args.append("--is_disconnected=True")
        warnings.warn(
            "No model / mode is given; chosen as default" + f"Using: {str_args[-1]}"
        )

    trainer: SingleTrainer = make_script_trainer(str_args=str_args, overrides=config)
    
    n_train = min(
        len(trainer.loaders[trainer.train_dataset_name]),
        trainer.config["optim"]["max_steps"],
    )
    print("Number of training batches:", n_train)
    print("Number of samples:", trainer.config["optim"]["batch_size"] * n_train)
    train_loader_iter = iter(trainer.loaders[trainer.train_dataset_name])
    i_for_epoch = 0

    for i in range(0, n_train):
        i_for_epoch += 1
        # Get a batch.
        batch = next(train_loader_iter)
        assert set(batch[0].atomic_numbers[batch[0].tags == 2].unique().tolist()).issubset({1, 8})

    print("DONE: only selected adsorbates are present in the dataset.")
