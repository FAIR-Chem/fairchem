import logging
import math
import sys
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal

import torch
import torch.nn as nn

from ocpmodels.common.flags import flags
from ocpmodels.common.utils import (
    build_config,
    new_trainer_context,
    setup_logging,
)
from ocpmodels.modules.scaling import ScaleFactor

if TYPE_CHECKING:
    from ocpmodels.trainers.base_trainer import BaseTrainer


def _train_batch(trainer: "BaseTrainer", batch):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=trainer.scaler is not None):
            out = trainer._forward(batch)
        loss = trainer._compute_loss(out, batch)
        del out, loss


def main(
    *,
    num_batches: int = 16,
):
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    _config = build_config(args, override_args)
    _config["logger"] = "tensorboard"

    assert not args.distributed, "This doesn't work with DDP"
    with new_trainer_context(args=args, config=_config) as ctx:
        config = ctx.config
        trainer = ctx.trainer

        ckpt_file = config.get("checkpoint", None)
        assert (
            ckpt_file is not None
        ), "Checkpoint file not specified. Please specify --checkpoint <path>"
        ckpt_file = Path(ckpt_file)

        logging.info(f"Run fitting for model: {args.identifier}")
        logging.info(f"Target ckpt path: {ckpt_file}")

        mode: Literal["all", "unfitted"] = "all"

        if ckpt_file.exists():
            logging.warning(f"Already found existing file: {ckpt_file}")
            flag = input(
                "Do you want to continue and fit all scale factors (1), "
                "only fit the variables not fitted yet (2), or exit (3)? "
            )
            if str(flag) == "1":
                mode = "all"
                logging.info("Fitting all scale factors.")
            elif str(flag) == "2":
                mode = "unfitted"
                logging.info("Only fitting unfitted variables.")
            else:
                print(flag)
                logging.info("Exiting script")
                sys.exit()

        model: nn.Module = trainer.model
        val_loader = trainer.val_loader
        assert (
            val_loader is not None
        ), "Val dataset is required for making predictions"

        if ckpt_file.exists():
            trainer.load_checkpoint(str(ckpt_file))

        model.eval()

        # recursively go through the submodules and get the ScaleFactor modules
        scale_factors: Dict[str, ScaleFactor] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, ScaleFactor)
        }

        if mode == "all":
            logging.info("Fitting all scale factors.")
            for name, scale_factor in scale_factors.items():
                if scale_factor.fitted.item():
                    logging.info(
                        f"{name} is already fitted in the checkpoint, resetting it."
                    )
                scale_factor.fitted[...] = False
                scale_factor.scale[...] = 1.0

        # we do a single pass through the network to get the correct execution order of the scale factors
        scale_factor_indices: Dict[str, int] = {}
        max_idx = 0

        def index_fn(module: ScaleFactor):
            nonlocal max_idx
            assert module.name is not None
            if module.name not in scale_factor_indices:
                scale_factor_indices[module.name] = max_idx
                logging.debug(f"Scale factor for {module.name} = {max_idx}")
                max_idx += 1

        # initialize all scale factors
        for name, scale_factor in scale_factors.items():
            scale_factor.initialize(name=name, index_fn=index_fn)

        # single pass through network
        _train_batch(trainer, next(iter(val_loader)))

        # sort the scale factors by their computation order
        sorted_factors = sorted(
            scale_factors.items(),
            key=lambda x: scale_factor_indices.get(x[0], math.inf),
        )

        logging.info("Sorted scale factors by computation order:")
        for name, _ in sorted_factors:
            logging.info(f"{name}: {scale_factor_indices[name]}")

        # loop over the scale factors in the computation order
        # and fit them one by one
        logging.info("Start fitting")

        for name, module in sorted_factors:
            if mode == "unfitted" and module.fitted.item():
                logging.info(f"Skipping {name} (already fitted)")
                continue

            with module.observe_and_fit():
                logging.info(f"Fitting {name}...")
                for batch in islice(val_loader, num_batches):
                    _train_batch(trainer, batch)

        # make sure all scale factors are fitted
        for name, module in sorted_factors:
            assert module.fitted.item(), f"{name} is not fitted"

        # save the scale factors to the checkpoint file
        trainer.config["cmd"]["checkpoint_dir"] = ckpt_file.parent
        trainer.is_debug = False
        out_file = trainer.save(
            metrics=None,
            checkpoint_file=f"{ckpt_file.stem}_scaled{ckpt_file.suffix}",
            training_state=False,
        )
        assert out_file is not None, "Failed to save checkpoint"
        out_file = Path(out_file)
        assert out_file.exists(), f"Failed to save checkpoint to {out_file}"
        logging.info(f"Saved results to: {out_file}")


if __name__ == "__main__":
    main()
