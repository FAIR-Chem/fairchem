"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import warnings
from ocpmodels.common.utils import make_script_trainer
from ocpmodels.trainers import EnergyTrainer
import sys

if __name__ == "__main__":
    config = {}
    # Customize args
    # config["model"]["energy_head"] = "weighted-av-initial-embeds"  # pooling, weighted-av-init-embeds, graclus, random # noqa: E501
    # config["frame_averaging"] = "2D"
    # config["model"]["graph_rewiring"] = "one-supernode-per-graph"
    # config["model"]["phys_embeds"] = True
    # config['model']['use_pbc'] = True
    config["model"]["graph_rewiring"] = "remove-tag-0"
    config["frame_averaging"] = "3d"
    config["choice_fa"] = "random"
    config["test_ri"] = True
    config["optim"]["max_epochs"] = 0
    config["model"]["use_pbc"] = False

    str_args = sys.argv[1:]
    if all("--config-yml" not in arg for arg in str_args):
        # str_args.append("--config-yml=configs/is2re/10k/dimenet_plus_plus/new_dpp.yml")
        # str_args.append("--config-yml=configs/is2re/10k/schnet/new_schnet.yml")
        # str_args.append("--config-yml=configs/is2re/10k/forcenet/new_forcenet.yml")
        str_args.append("--config-yml=configs/is2re/10k/sfarinet/sfarinet.yml")
        # str_args.append("--config-yml=configs/is2re/10k/fanet/fanet.yml")
        # str_args.append("--checkpoint=checkpoints/2022-04-26-12-23-28-schnet/best_checkpoint.pt")
        warnings.warn(
            "No model / mode is given; chosen as default" + f"Using: {str_args[-1]}"
        )

    trainer: EnergyTrainer = make_script_trainer(str_args=str_args, overrides=config)

    trainer.train()

    trainer.load_checkpoint(
        checkpoint_path="checkpoints/2022-04-28-11-42-56-dimenetplusplus/"
        + "best_checkpoint.pt"
    )

    predictions = trainer.predict(
        trainer.val_loader, results_file="is2re_results", disable_tqdm=False
    )
