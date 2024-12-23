from __future__ import annotations

import logging
import math
import re
import readline
import sys
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel

from fairchem.core.common.flags import flags
from fairchem.core.common.utils import build_config, new_trainer_context, setup_logging
from fairchem.core.modules.scaling import ScaleFactor
from fairchem.core.modules.scaling.compat import load_scales_compat

if TYPE_CHECKING:
    from fairchem.core.trainers.base_trainer import BaseTrainer


def _prefilled_input(prompt: str, prefill: str = "") -> str:
    readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()


def _train_batch(trainer: BaseTrainer, batch) -> None:
    with torch.no_grad():
        with torch.autocast("cuda", enabled=trainer.scaler is not None):
            out = trainer._forward(batch)
        loss = trainer._compute_loss(out, batch)
        del out, loss


def compute_scaling_factors(config, num_batches: int = 16) -> None:
    with new_trainer_context(config=config) as ctx:
        config = ctx.config
        trainer = ctx.trainer

        ckpt_file = config.get("checkpoint", None)
        assert (
            ckpt_file is not None
        ), "Checkpoint file not specified. Please specify --checkpoint <path>"
        ckpt_file = Path(ckpt_file)

        logging.info(f"Input checkpoint path: {ckpt_file}, {ckpt_file.exists()=}")

        model: nn.Module = trainer.model
        data_loader = trainer.train_loader
        assert data_loader is not None, "Train set required to load batches"

        if ckpt_file.exists():
            trainer.load_checkpoint(checkpoint_path=str(ckpt_file))

        # region reoad scale file contents if necessary
        # unwrap module from DP/DDP
        unwrapped_model = model
        while isinstance(unwrapped_model, DistributedDataParallel):
            unwrapped_model = unwrapped_model.module
        assert isinstance(unwrapped_model, nn.Module), "Model is not a nn.Module"
        load_scales_compat(unwrapped_model, config.get("scale_file", None))
        # endregion

        model.eval()

        # recursively go through the submodules and get the ScaleFactor modules
        scale_factors: dict[str, ScaleFactor] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, ScaleFactor)
        }

        mode: Literal["all", "unfitted"] = "all"

        # region detect fitted/unfitted factors
        fitted_scale_factors = [
            f"{name}: {module.scale_factor.item():.3f}"
            for name, module in scale_factors.items()
            if module.fitted
        ]
        unfitted_scale_factors = [
            name for name, module in scale_factors.items() if not module.fitted
        ]
        fitted_scale_factors_str = ", ".join(fitted_scale_factors)
        logging.info(f"Fitted scale factors: [{fitted_scale_factors_str}]")
        unfitted_scale_factors_str = ", ".join(unfitted_scale_factors)
        logging.info(f"Unfitted scale factors: [{unfitted_scale_factors_str}]")

        if fitted_scale_factors:
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
            elif str(flag) == "3":
                logging.info("Exiting script")
                sys.exit()
            else:
                logging.error(
                    f"Unrecognized flag associated with fitted_scale_factors: '{flag}'. Exiting."
                )
                sys.exit(-1)
        # endregion

        if ckpt_file.exists():
            logging.warning(f"Already found existing file: {ckpt_file}")
            flag = input(
                "Do you want to continue and overwrite existing file (1), "
                "or exit (2)? "
            )
            if str(flag) == "1":
                logging.info("Overwriting existing file.")
            else:
                logging.info("Exiting script")
                sys.exit()

        logging.info(
            f"Output path for fitted scale factors: {ckpt_file}, {ckpt_file.exists()=}"
        )
        # endregion

        # region reset the scale factors if mode == "all"
        if mode == "all":
            logging.info("Fitting all scale factors.")
            for name, scale_factor in scale_factors.items():
                if scale_factor.fitted:
                    logging.info(
                        f"{name} is already fitted in the checkpoint, resetting it. {scale_factor.scale_factor}"
                    )
                scale_factor.reset_()
        # endregion

        # region we do a single pass through the network to get the correct execution order of the scale factors
        scale_factor_indices: dict[str, int] = {}
        max_idx = 0

        # initialize all scale factors
        for name, module in scale_factors.items():

            def index_fn(name: str = name) -> None:
                nonlocal max_idx
                assert name is not None
                if name not in scale_factor_indices:
                    scale_factor_indices[name] = max_idx
                    logging.debug(f"Scale factor for {name} = {max_idx}")
                    max_idx += 1

            module.initialize_(index_fn=index_fn)

        # single pass through network
        _train_batch(trainer, next(iter(data_loader)))

        # sort the scale factors by their computation order
        sorted_factors = sorted(
            scale_factors.items(),
            key=lambda x: scale_factor_indices.get(x[0], math.inf),
        )

        logging.info("Sorted scale factors by computation order:")
        for name, _ in sorted_factors:
            logging.info(f"{name}: {scale_factor_indices[name]}")

        # endregion

        # loop over the scale factors in the computation order
        # and fit them one by one
        logging.info("Start fitting")

        for name, module in sorted_factors:
            if mode == "unfitted" and module.fitted:
                logging.info(f"Skipping {name} (already fitted)")
                continue

            logging.info(f"Fitting {name}...")
            with module.fit_context_():
                for batch in islice(data_loader, num_batches):
                    _train_batch(trainer, batch)
                stats, ratio, value = module.fit_()

                logging.info(
                    f"Variable: {name}, "
                    f"Var_in: {stats['variance_in']:.3f}, "
                    f"Var_out: {stats['variance_out']:.3f}, "
                    f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
                )

        # make sure all scale factors are fitted
        for name, module in sorted_factors:
            assert module.fitted, f"{name} is not fitted"

        # region save the scale factors to the checkpoint file
        trainer.config["cmd"]["checkpoint_dir"] = ckpt_file.parent
        trainer.is_debug = False

        def rename_module(name):
            name = name.replace(".scale_factor", "")
            # remove DDP wrapper
            name = re.sub("^module.", "", name)
            # remove hydra backbone
            return re.sub("^backbone.", "", name)

        torch.save(
            {
                rename_module(x[0]): x[1]
                for x in trainer.model.to("cpu").named_parameters()
                if ".scale_" in x[0]
            },
            str(ckpt_file),
        )
        logging.info(f"Saved results to: {ckpt_file}")


if __name__ == "__main__":
    # region args/config setup
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    assert not args.distributed, "This doesn't work with DDP"
    config = build_config(args, override_args)

    compute_scaling_factors(config)
