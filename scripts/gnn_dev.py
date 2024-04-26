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
    config["graph_rewiring"] = ""
    config["frame_averaging"] = "2D"
    config["fa_method"] = "random"  # "random"
    config["test_ri"] = False
    config["optim"] = {"max_epochs": 1}
    config["model"] = {"use_pbc": True}

    checkpoint_path = None
    # "checkpoints/2022-04-28-11-42-56-dimenetplusplus/" + "best_checkpoint.pt"

    str_args = sys.argv[1:]
    if all("config" not in arg for arg in str_args):
        str_args.append("--is_debug")
        str_args.append("--config=deup_depfaenet-deup_is2re-10k")
        # str_args.append("--adsorbates={'*O', '*OH', '*OH2', '*H'}")
        str_args.append("--is_disconnected=True")
        # str_args.append("--silent=0")
        warnings.warn(
            "No model / mode is given; chosen as default" + f"Using: {str_args[-1]}"
        )

    trainer: SingleTrainer = make_script_trainer(str_args=str_args, overrides=config)

    trainer.train()

    if checkpoint_path:
        trainer.load_checkpoint(
            checkpoint_path="checkpoints/2022-04-28-11-42-56-dimenetplusplus/"
            + "best_checkpoint.pt"
        )

        predictions = trainer.predict(
            trainer.val_loader, results_file="is2re_results", disable_tqdm=False
        )
