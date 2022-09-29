"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import warnings

from ocpmodels.common.flags import flags
from ocpmodels.common.utils import build_config, setup_imports
from ocpmodels.trainers import EnergyTrainer

if __name__ == "__main__":

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()

    if not args.mode or not args.config_yml:
        args.mode = "train"
        # args.config_yml = "configs/is2re/10k/dimenet_plus_plus/new_dpp.yml"
        # args.config_yml = "configs/is2re/10k/schnet/new_schnet.yml"
        # args.config_yml = "configs/is2re/10k/forcenet/new_forcenet.yml"
        args.config_yml = "configs/is2re/10k/sfarinet/sfarinet.yml"
        # args.config_yml = "configs/is2re/10k/fanet/fanet.yml"
        # args.checkpoint = "checkpoints/2022-04-26-12-23-28-schnet/best_checkpoint.pt"
        warnings.warn("No model / mode is given; chosen as default")

    config = build_config(args, override_args)

    # Customise args
    # config["model"]["energy_head"] = "weighted-av-initial-embeds"  # pooling, weighted-av-init-embeds, graclus, random
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

    setup_imports()
    trainer = EnergyTrainer(
        task=config["task"],
        model_attributes=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        run_dir=config["run_dir"],
        is_debug=True,
        print_every=100,
        seed=config["seed"],
        logger=config["logger"],
        local_rank=config["local_rank"],
        amp=config["amp"],
        cpu=config["cpu"],
        new_gnn=config.get("new_gnn"),
        frame_averaging=config["frame_averaging"],
        test_ri=config["test_ri"],
        choice_fa=config["choice_fa"],
    )

    trainer.train()

    trainer.load_checkpoint(
        checkpoint_path="checkpoints/2022-04-28-11-42-56-dimenetplusplus/best_checkpoint.pt"
    )

    predictions = trainer.predict(
        trainer.val_loader, results_file="is2re_results", disable_tqdm=False
    )
