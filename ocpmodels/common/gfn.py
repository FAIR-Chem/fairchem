from copy import deepcopy
from pathlib import Path
from typing import Callable, Union, List

import os

import torch.nn as nn
from torch_geometric.data.data import Data
from torch_geometric.data.batch import Batch

from ocpmodels.common.utils import make_trainer_from_dir, resolve
from ocpmodels.models.faenet import FAENet
from ocpmodels.datasets.data_transforms import get_transforms


class FAENetWrapper(nn.Module):
    def __init__(
        self,
        faenet: FAENet,
        transform: Callable = None,
        frame_averaging: str = None,
        trainer_config: dict = None,
    ):
        """
        `FAENetWrapper` is a wrapper class for the FAENet model. It is used to perform
        a forward pass of the model when frame averaging is applied.

        Args:
            faenet (FAENet, optional): The FAENet model to use. Defaults to None.
            transform (Transform, optional): The data transform to use. Defaults to None.
            frame_averaging (str, optional): The frame averaging method to use.
            trainer_config (dict, optional): The trainer config used to create the model.
                Defaults to None.
        """
        super().__init__()

        self.faenet = faenet
        self.transform = transform
        self.frame_averaging = frame_averaging
        self.trainer_config = trainer_config
        self._is_frozen = None

    @property
    def frozen(self):
        """
        Returns whether or not the model is frozen. A model is frozen if all of its
        parameters are set to not require gradients.

        This is a lazy property, meaning that it is only computed once and then cached.

        Returns:
            bool: Whether or not the model is frozen.
        """
        if self._is_frozen is None:
            frozen = True
            for param in self.parameters():
                if param.requires_grad:
                    frozen = False
                    break
            self._is_frozen = frozen
        return self._is_frozen

    def preprocess(self, batch: Union[Batch, Data, List[Data], List[Batch]]):
        """
        Preprocess a batch of graphs using the data transform.

        * if batch is a list with one element:
            * it could be a batch from the FAENet data loader which produces
                lists of Batch with 1 element (because of multi-GPU features)
            * if the single element is a Batch, extract it (`batch=batch[0]`)
        * if batch is a Data instance, it is a single graph and we turn
            it back into a list of 1 element (`batch=[batch]`)
        * if it is a Batch instance, it is a collection of graphs and we turn it
            into a list of Data graphs (`batch=batch.to_data_list()`)

        Finally we transform the list of Data graphs with the pre-processing transforms
        and collate them into a Batch.

        .. code-block:: python

            In [7]: %timeit wrapper.preprocess(batch)
            The slowest run took 4.94 times longer than the fastest.
            This could mean that an intermediate result is being cached.
            67.1 ms ± 58.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            In [8]: %timeit wrapper.preprocess(batch)
            43.8 ms ± 1.66 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        Args:
            batch (List?[Data, Batch]): The batch of graphs to transform

        Returns:
            torch_geometric.Batch: The transformed batch. If frame averaging is
                disabled, this is the same as the input batch.
        """
        if isinstance(batch, list):
            if len(batch) == 1 and isinstance(batch[0], Batch):
                batch = batch[0]
        if isinstance(batch, Data):
            batch = [batch]
        if isinstance(batch, Batch):
            batch = batch.to_data_list()

        return Batch.from_data_list([self.transform(b) for b in batch])

    def forward(
        self,
        batch: Union[Batch, Data, List[Data], List[Batch]],
        preprocess: bool = True,
        retrieve_hidden: bool = False,
    ):
        """Perform a forward pass of the model when frame averaging is applied.

        Adapted from
        ocmpodels.trainers.single_point_trainer.SingleTrainer.model_forward

        This implementation assumes only the energy is being predicted, and only
        frame-averages this prediction.

        Args:
            batch (List?[Data, Batch]): The batch of graphs to predict on.
            preprocess (bool, optional): Whether or not to apply the data transforms.
                Defaults to True.

        Returns:
            (dict): model predictions tensor for "energy" and "forces".
        """
        if preprocess:
            batch = self.preprocess(batch)
        if not self.frozen:
            raise RuntimeError(
                "FAENetWrapper must be frozen before calling forward."
                + " Use .freeze() to freeze it."
            )
        # Distinguish frame averaging from base case.
        if self.frame_averaging and self.frame_averaging != "DA":
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
                    mode="inference",
                    regress_forces=False,
                    q=None,
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

        if retrieve_hidden:
            return preds
        return preds["energy"]  # denormalize?

    def freeze(self):
        """Freeze the model parameters."""
        for param in self.parameters():
            param.requires_grad = False


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

    The loaded model is frozen (all parameters are set to not require gradients).

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
        overrides={
            "is_debug": True,
            "silent": True,
            "cp_data_to_tmpdir": False,
        },
        silent=True,
    )

    wrapper = FAENetWrapper(
        faenet=trainer.model,
        transform=get_transforms(trainer.config),
        frame_averaging=trainer.config.get("frame_averaging", ""),
        trainer_config=trainer.config,
    )
    wrapper.freeze()
    loaders = trainer.loaders

    return wrapper, loaders


if __name__ == "__main__":
    # for instance in ipython:
    # In [1]: run ocpmodels/common/gfn.py
    #
    from ocpmodels.common.gfn import prepare_for_gfn

    ckpt_paths = {"mila": "/path/to/releases_dir"}
    release = "v2.3_graph_phys"
    # or
    ckpt_paths = {
        "mila": "/network/scratch/s/schmidtv/ocp/runs/3789733/checkpoints/best_checkpoint.pt"
    }
    release = None
    wrapper, loaders = prepare_for_gfn(ckpt_paths, release)
    data_gen = iter(loaders["train"])
    batch = next(data_gen)
    preds = wrapper(batch)
