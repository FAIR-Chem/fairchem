from copy import deepcopy
from pathlib import Path
from typing import Union

import os

import torch.nn as nn

from ocpmodels.common.utils import make_trainer_from_dir, merge_dicts, resolve
from ocpmodels.models.faenet import FAENet


class FAENetWrapper(nn.Module):
    def __init__(
        self,
        config: dict = {},
        config_path: str = None,
        overrides: dict = {},
        silent=True,
    ):
        """
        `FAENetWrapper` is a wrapper class for the FAENet model. It is used to perform
        a forward pass of the model when frame averaging is applied.

        You must provide exactly one of `config` or `config_path`. If `config` is
        provided, it must contain a `frame_averaging` key and a `model` section. If
        `config_path` is provided, it must point to either a single `.ckpt` file or to
        a directory with checkpoints as `checkpoints/checkpoint-*.pt`.


        Args:
            config (dict, optional): Wrapper and model config, akin to a Trainer's
                config. In particular it must contain a `frame_averaging` key and a
                `model` section. Defaults to {}.
            config_path (str, optional): Path to either a single `.ckpt` file or to
                a directory with checkpoints as `checkpoints/checkpoint-*.pt`. Defaults
                to None.
            overrides (dict, optional): A dictionary to override configs before loading
                the model. Defaults to {}.
            silent (bool, optional): Whether or not loading should be silent, i.e. not
                print anything. Defaults to True.
        """
        assert config_path or config, "Either config or config_path must be provided."
        assert not (
            config_path and config
        ), "Only one of config or config_path must be provided."

        super().__init__()

        if not config:
            trainer = make_trainer_from_dir(
                config_path,
                mode="continue",
                overrides=overrides,
                silent=silent,
            )
            config = trainer.config
        else:
            config = merge_dicts(config, overrides)

        self.config = config
        assert "model" in self.config, "Config must contain a model section."

    def setup_model(self):
        self.faenet = FAENet(**self.config["model"])

    def forward(self, batch, mode="inference", q=None):
        """Perform a forward pass of the model when frame averaging is applied.

        Adapted from
        ocmpodels.trainers.single_point_trainer.SingleTrainer.model_forward raz rza

        Returns:
            (dict): model predictions tensor for "energy" and "forces".
        """
        # Distinguish frame averaging from base case.
        if self.config["frame_averaging"] and self.config["frame_averaging"] != "DA":
            original_pos = batch[0].pos
            original_cell = batch[0].cell
            e_all = []

            # Compute model prediction for each frame
            for i in range(len(batch[0].fa_pos)):
                batch[0].pos = batch[0].fa_pos[i]
                batch[0].cell = batch[0].fa_cell[i]

                # forward pass
                preds = self.faenet(
                    deepcopy(batch),
                    mode=mode,
                    regress_forces=False,
                    q=q,
                )
                e_all.append(preds["energy"])

            batch[0].pos = original_pos
            batch[0].cell = original_cell

            # Average predictions over frames
            preds["energy"] = sum(e_all) / len(e_all)
        else:
            preds = self.faenet(batch)

        if preds["energy"].shape[-1] == 1:
            preds["energy"] = preds["energy"].view(-1)

        return preds


def parse_loc() -> str:
    """
    Parse the current location from the environment variables. If the location is a
    number, assume it is a SLURM job ID and return "mila". Otherwise, return the
    location name.

    Returns:
        str: Where the current job is running, typically Mila or DRAC or laptop.
    """
    loc = os.environ.get(
        "SLURM_CLUSTER_NAME", os.environ.get("SLURM_JOB_ID", os.environ["USER"])
    )
    if all(s.isdigit() for s in loc):
        loc = "mila"
    return loc


def find_ckpt(ckpt_paths: dict, release: str) -> Path:
    """
    Finds a checkpoint in a dictionary of paths, based on the current cluster name and
    release. If the path is a file, use it directly. Otherwise, look for a single
    checkpoint file in a ${release}/sub-fodler. E.g.:
        ckpt_paths = {"mila": "/path/to/ckpt_dir"} release = v2.3_graph_phys
        find_ckpt(ckpt_paths, release) -> /path/to/ckpt_dir/v2.3_graph_phys/name.ckpt

        ckpt_paths = {"mila": "/path/to/ckpt_dir/file.ckpt"} release = v2.3_graph_phys
        find_ckpt(ckpt_paths, release) -> /path/to/ckpt_dir/file.ckpt

    Args:
        ckpt_paths (dict): Where to look for the checkpoints.
            Maps cluster names to paths.

    Raises:
        ValueError: The current location is not in the checkpoint path dict.
        ValueError: The checkpoint path does not exist. ValueError: The checkpoint path
        is a directory and contains no .ckpt file. ValueError: The checkpoint path is a
        directory and contains >1 .ckpt files.

    Returns:
        Path: Path to the checkpoint for that release on this host.
    """
    loc = parse_loc()
    if loc not in ckpt_paths:
        raise ValueError(f"FAENet proxy checkpoint path not found for location {loc}.")
    path = resolve(ckpt_paths[loc])
    if not path.exists():
        raise ValueError(f"FAENet proxy checkpoint not found at {str(path)}.")
    if path.is_file():
        return path
    path = path / release
    ckpts = list(path.glob("**/*.ckpt"))
    if len(ckpts) == 0:
        raise ValueError(f"No FAENet proxy checkpoint found at {str(path)}.")
    if len(ckpts) > 1:
        raise ValueError(
            f"Multiple FAENet proxy checkpoints found at {str(path)}. "
            "Please specify the checkpoint explicitly."
        )
    return ckpts[0]


def prepare_for_gfn(ckpt_paths: dict, release: str) -> tuple:
    """
    Prepare a FAENet model for use in GFN. Loads the checkpoint for the given release
    on the current host, and wraps it in a FAENetWrapper.

    Example ckpt_paths:

    ckpt_paths = {
        "mila": "/path/to/releases_dir",
        "drac": "/path/to/releases_dir",
        "laptop": "/path/to/releases_dir",
    }

    Args:
        ckpt_paths (dict): Where to look for the checkpoints as {loc: path}.
        release (str): Which release to load.

    Returns:
        tuple: (model, loaders) where loaders is a dict of loaders for the model.
    """
    ckpt_path = find_ckpt(ckpt_paths, release)
    assert ckpt_path.exists(), f"Path {ckpt_path} does not exist."
    trainer = make_trainer_from_dir(
        ckpt_path,
        mode="continue",
        overrides={},
        silent=True,
    )

    model = FAENetWrapper(config=trainer.config)
    loaders = trainer.loaders

    return model, loaders
