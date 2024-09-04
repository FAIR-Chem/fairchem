"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fairchem.core.datasets import data_list_collater

from ._load_utils import _load_from_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fairchem.core.modules.normalization.element_references import LinearReferences


class Normalizer(nn.Module):
    """Normalize/denormalize a tensor and optionally add a atom reference offset."""

    def __init__(
        self,
        mean: float | torch.Tensor = 0.0,
        rmsd: float | torch.Tensor = 1.0,
    ):
        """tensor is taken as a sample to calculate the mean and rmsd"""
        super().__init__()

        if isinstance(mean, float):
            mean = torch.tensor(mean)
        if isinstance(rmsd, float):
            rmsd = torch.tensor(rmsd)

        self.register_buffer(name="mean", tensor=mean)
        self.register_buffer(name="rmsd", tensor=rmsd)

    @torch.autocast(device_type="cuda", enabled=False)
    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.rmsd

    @torch.autocast(device_type="cuda", enabled=False)
    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return normed_tensor * self.rmsd + self.mean

    def forward(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return self.denorm(normed_tensor)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        # check if state dict is legacy state dicts
        if "std" in state_dict:
            state_dict = {
                "mean": torch.tensor(state_dict["mean"]),
                "rmsd": state_dict["std"],
            }

        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def create_normalizer(
    file: str | Path | None = None,
    state_dict: dict | None = None,
    tensor: torch.Tensor | None = None,
    mean: float | torch.Tensor | None = None,
    rmsd: float | torch.Tensor | None = None,
    stdev: float | torch.Tensor | None = None,
) -> Normalizer:
    """Build a target data normalizers with optional atom ref

    Only one of file, state_dict, tensor, or (mean and rmsd) will be used to create a normalizer.
    If more than one set of inputs are given priority will be given following the order in which they are listed above.

    Args:
        file (str or Path): path to pt or npz file.
        state_dict (dict): a state dict for Normalizer module
        tensor (Tensor): a tensor with target values used to compute mean and std
        mean (float | Tensor): mean of target data
        rmsd (float | Tensor): rmsd of target data, rmsd from mean = stdev, rmsd from 0 = rms
        stdev: standard deviation (deprecated, use rmsd instead)

    Returns:
        Normalizer
    """
    if stdev is not None:
        warnings.warn(
            "Use of 'stdev' is deprecated, use 'rmsd' instead", DeprecationWarning
        )
        if rmsd is not None:
            logging.warning(
                "Both 'stdev' and 'rmsd' values where given to create a normalizer, rmsd values will be used."
            )

    # old configs called it stdev, using this in the function signature reduces overhead code elsewhere
    if stdev is not None and rmsd is None:
        rmsd = stdev

    # path takes priority if given
    if file is not None:
        if state_dict is not None or tensor is not None or mean is not None:
            logging.warning(
                "A file to a normalizer has been given. Normalization values will be read from it, and all other inputs"
                " will be ignored."
            )
        extension = Path(file).suffix
        if extension == ".pt":
            # try to load a pt file
            state_dict = torch.load(file)
        elif extension == ".npz":
            # try to load an NPZ file
            values = np.load(file)
            mean = values.get("mean")
            rmsd = values.get("rmsd") or values.get("std")  # legacy files
            tensor = None  # set to None since values read from file are prioritized
        else:
            raise RuntimeError(
                f"Normalizer file with extension '{extension}' is not supported."
            )

    # state dict is second priority
    if state_dict is not None:
        if tensor is not None or mean is not None:
            logging.warning(
                "The state_dict provided will be used to set normalization values. All other inputs will be ignored."
            )
        normalizer = Normalizer()
        normalizer.load_state_dict(state_dict)
        return normalizer

    # if not then read target value tensor
    if tensor is not None:
        if mean is not None:
            logging.warning(
                "Normalization values will be computed from input tensor, all other inputs will be ignored."
            )
        mean = torch.mean(tensor)
        rmsd = torch.std(tensor)
    elif mean is not None and rmsd is not None:
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(rmsd, torch.Tensor):
            rmsd = torch.tensor(rmsd)

    # if mean and rmsd are still None than raise an error
    if mean is None or rmsd is None:
        raise ValueError(
            "Incorrect inputs. One of the following sets of inputs must be given: ",
            "a file path to a .pt or .npz file, or mean and rmsd values, or a tensor of target values",
        )

    return Normalizer(mean=mean, rmsd=rmsd)


@torch.no_grad()
def fit_normalizers(
    targets: list[str],
    dataset: Dataset,
    batch_size: int,
    override_values: dict[str, dict[str, float]] | None = None,
    rmsd_correction: int | None = None,
    element_references: dict | None = None,
    num_batches: int | None = None,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 0,
) -> dict[str, Normalizer]:
    """Estimate mean and rmsd from data to create normalizers

    Args:
        targets: list of target names
        dataset: data set to fit linear references with
        batch_size: size of batch
        override_values: dictionary with target names and values to override. i.e. {"forces": {"mean": 0.0}} will set
            the forces mean to zero.
        rmsd_correction: correction to use when computing mean in std/rmsd. See docs for torch.std.
            If not given, will always use 0 when mean == 0, and 1 otherwise.
        element_references:
        num_batches: number of batches to use in fit. If not given will use all batches
        num_workers: number of workers to use in data loader
            Note setting num_workers > 1 leads to finicky multiprocessing issues when using this function
            in distributed mode. The issue has to do with pickling the functions in load_normalizers_from_config
            see function below...
        shuffle: whether to shuffle when loading the dataset
        seed: random seed used to shuffle the sampler if shuffle=True

    Returns:
        dict of normalizer objects
    """
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(data_list_collater, otf_graph=True),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )

    num_batches = num_batches if num_batches is not None else len(data_loader)
    if num_batches > len(data_loader):
        logging.warning(
            f"The given num_batches {num_batches} is larger than total batches of size {batch_size} in dataset. "
            f"num_batches will be ignored and the whole dataset will be used."
        )
        num_batches = len(data_loader)

    element_references = element_references or {}
    target_vectors = defaultdict(list)

    logging.info(
        f"Estimating mean and rmsd for normalization using {num_batches * batch_size} samples in {num_batches} batches "
        f"of size {batch_size}."
    )
    for i, batch in tqdm(
        enumerate(data_loader), total=num_batches, desc="Estimating mean and rmsd"
    ):
        if i == num_batches:
            break

        for target in targets:
            target_vector = batch[target]
            if target in element_references:
                target_vector = element_references[target].dereference(
                    target_vector, batch, reshaped=False
                )
            target_vectors[target].append(target_vector)

    normalizers = {}
    for target in targets:
        target_vector = torch.cat(target_vectors[target], dim=0)
        values = {"mean": target_vector.mean()}
        if target in override_values:
            for name, val in override_values[target].items():
                values[name] = torch.tensor(val)
        # calculate root mean square deviation
        if "rmsd" not in values:
            if rmsd_correction is None:
                rmsd_correction = 0 if values["mean"] == 0.0 else 1
            values["rmsd"] = (
                ((target_vector - values["mean"]) ** 2).sum()
                / max(len(target_vector) - rmsd_correction, 1)
            ).sqrt()
        normalizers[target] = create_normalizer(**values)

    return normalizers


def load_normalizers_from_config(
    config: dict[str, Any],
    dataset: Dataset,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
    element_references: dict[str, LinearReferences] | None = None,
) -> dict[str, Normalizer]:
    """Create a dictionary with element references from a config."""
    # edit the config slightly to extract override args
    if "fit" in config:  # noqa
        if "override_values" not in config["fit"] and isinstance(
            config["fit"]["targets"], dict
        ):
            override_values = {
                target: vals
                for target, vals in config["fit"]["targets"].items()
                if isinstance(vals, dict)
            }
            config["fit"]["override_values"] = override_values
            config["fit"]["targets"] = list(config["fit"]["targets"].keys())

    return _load_from_config(
        config,
        "normalizers",
        fit_normalizers,
        create_normalizer,
        dataset,
        checkpoint_dir,
        seed=seed,
        element_references=element_references,
    )
