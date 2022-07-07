"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import sys

import torch
from tqdm import tqdm

from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import build_config, setup_imports, setup_logging

if __name__ == "__main__":
    setup_logging()

    num_batches = 16  # number of batches to use to fit a single variable

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    assert config["model"]["name"].startswith("gemnet")
    config["logger"] = "tensorboard"

    if args.distributed:
        raise ValueError(
            "I don't think this works with DDP (race conditions)."
        )

    setup_imports()

    scale_file = config["model"]["scale_file"]
    config["model"]["scale_file"] = None

    logging.info(f"Run fitting for model: {config['model']['name']}")
    logging.info(f"Target scale file: {scale_file}")

    trainer = registry.get_trainer_class(config.get("trainer", "simple"))(
        task=config["task"],
        model=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        run_dir=config.get("run_dir", "./"),
        is_debug=config.get("is_debug", False),
        is_vis=config.get("is_vis", False),
        print_every=config.get("print_every", 10),
        seed=config.get("seed", 0),
        logger=config.get("logger", "tensorboard"),
        local_rank=config["local_rank"],
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
        slurm=config.get("slurm", {}),
    )

    fit_factors = dict(trainer.model.module.scaling_factors())

    if os.path.exists(scale_file):
        logging.warning(f"Already found existing file: {scale_file}")
        flag = input(
            "Do you want to continue and overwrite the file (1), "
            "only fit the variables not fitted yet (2), or exit (3)? "
        )
        if str(flag) == "1":
            logging.info("Overwriting the current file.")
        elif str(flag) == "2":
            logging.info("Only fitting unfitted variables.")
            old_scales = torch.load(scale_file)
            incompatible_factors = trainer.model.module.load_scales(
                old_scales, strict=False
            )
            missing_factors = incompatible_factors.missing_factors
            fit_factors = {k: fit_factors[k] for k in missing_factors}
        else:
            logging.info("Exiting script")
            sys.exit()

    # Fitting loop
    # The factors are fit in the order they are initialized in the model.
    # They should thus be added as object attributes in the right order
    # (earlier used factors first) to ensure stability.
    logging.info("Start fitting")

    assert (
        trainer.val_loader is not None
    ), "Val dataset is required for making predictions"
    with torch.no_grad():
        trainer.model.eval()
        for name, scaler in tqdm(fit_factors.items()):
            scaler.start_fitting()
            for i, batch in enumerate(trainer.val_loader):
                with torch.cuda.amp.autocast(
                    enabled=trainer.scaler is not None
                ):
                    out = trainer._forward(batch)
                loss = trainer._compute_loss(out, batch)
                del out, loss
                if i == num_batches:
                    break

            logging.info(f"Scaling factor '{name}':")
            scaler.finalize_fitting()

    torch.save(
        {k: v.scale_factor for k, v in trainer.model.module.scaling_factors()},
        scale_file,
    )
    logging.info(f"Fitting done. Results saved to: {scale_file}")
