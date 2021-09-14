"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

Script for calculating the scaling factors used to even out GemNet activation
scales. This generates the `scale_file` specified in the config, which is then
read in at model initialization.
This only needs to be run if the hyperparameters or model change
in places were it would affect the activation scales.
"""

import logging
import os
import sys

import torch
from tqdm import trange

from ocpmodels.models.gemnet.layers.scaling import AutomaticFit
from ocpmodels.models.gemnet.utils import write_json
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

    logging.info(f"Run fitting for model: {args.identifier}")
    logging.info(f"Target scale file: {scale_file}")

    def initialize_scale_file(scale_file):
        # initialize file
        preset = {"comment": args.identifier}
        write_json(scale_file, preset)

    if os.path.exists(scale_file):
        logging.warning(f"Already found existing file: {scale_file}")
        flag = input(
            "Do you want to continue and overwrite the file (1), "
            "only fit the variables not fitted yet (2), or exit (3)? "
        )
        if str(flag) == "1":
            logging.info("Overwriting the current file.")
            initialize_scale_file(scale_file)
        elif str(flag) == "2":
            logging.info("Only fitting unfitted variables.")
        else:
            print(flag)
            logging.info("Exiting script")
            sys.exit()
    else:
        initialize_scale_file(scale_file)

    AutomaticFit.set2fitmode()

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

    # Fitting loop
    logging.info("Start fitting")

    if not AutomaticFit.fitting_completed():
        with torch.no_grad():
            trainer.model.eval()
            for _ in trange(len(AutomaticFit.queue) + 1):
                assert (
                    trainer.val_loader is not None
                ), "Val dataset is required for making predictions"

                for i, batch in enumerate(trainer.val_loader):
                    with torch.cuda.amp.autocast(
                        enabled=trainer.scaler is not None
                    ):
                        out = trainer._forward(batch)
                    loss = trainer._compute_loss(out, batch)
                    del out, loss
                    if i == num_batches:
                        break

                current_var = AutomaticFit.activeVar
                if current_var is not None:
                    current_var.fit()  # fit current variable
                else:
                    print("Found no variable to fit. Something went wrong!")

    assert AutomaticFit.fitting_completed()
    logging.info(f"Fitting done. Results saved to: {scale_file}")
