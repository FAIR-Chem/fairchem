"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
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

    from fairchem.core.modules.normalization.element_references import LinearReference


class Normalizer(nn.Module):
    """Normalize/denormalize a tensor and optionally add a atom reference offset."""

    def __init__(
        self,
        mean: float | torch.Tensor = 0.0,
        std: float | torch.Tensor = 1.0,
    ):
        """tensor is taken as a sample to calculate the mean and std"""
        super().__init__()

        if isinstance(mean, float):
            mean = torch.tensor(mean)
        if isinstance(std, float):
            std = torch.tensor(std)

        self.register_buffer(name="mean", tensor=mean)
        self.register_buffer(name="std", tensor=std)

    @torch.autocast(device_type="cuda", enabled=False)
    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    @torch.autocast(device_type="cuda", enabled=False)
    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return normed_tensor * self.std + self.mean

    def forward(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return self.denorm(normed_tensor)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        # check if state dict is legacy state dicts
        if isinstance(state_dict["mean"], float):
            state_dict.update({k: torch.tensor(state_dict[k]) for k in ("mean", "std")})

        super().load_state_dict(state_dict, strict=strict, assign=assign)


def create_normalizer(
    file: str | Path | None = None,
    state_dict: dict | None = None,
    tensor: torch.Tensor | None = None,
    mean: float | torch.Tensor | None = None,
    stdev: float | torch.Tensor | None = None,
) -> Normalizer:
    """Build a target data normalizers with optional atom ref

    Only one of file, state_dict, tensor, or (mean and std) will be used to create a normalizer.
    If more than one set of inputs are given priority will be given following the order in which they are listed above.

    Args:
        file (str or Path): path to pt or npz file.
        state_dict (dict): a state dict for Normalizer module
        tensor (Tensor): a tensor with target values used to compute mean and std
        mean (float | Tensor): mean of target data
        stdev (float | Tensor): std of target data
        data_loader (DataLoader): a data loader used to estimate mean and std
        num_batches (int): the number of batches used. If not given all batches are used.

    Returns:
        Normalizer
    """
    std = stdev  # old configs called it stdev, using this in the function signature reduces overhead code elsewhere
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
            std = values.get("std")
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
        std = torch.std(tensor)
    elif mean is not None and std is not None:
        mean = torch.tensor(mean)
        std = torch.tensor(std)

    # if mean and std are still None than raise an error
    if mean is None or std is None:
        raise ValueError(
            "Incorrect inputs. One of the following sets of inputs must be given: ",
            "a file path to a .pt or .npz file, or mean and std values, or a tensor of target values",
        )

    return Normalizer(mean=mean, std=std)


def fit_normalizers(
    targets: list[str],
    dataset: Dataset,
    batch_size: int,
    element_references: dict | None = None,
    num_batches: int | None = None,
    num_workers: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> dict[str, Normalizer]:
    """Estimate mean and std from data to create normalizers

    Args:
        targets: list of target names
        dataset: data set to fit linear references with
        batch_size: size of batch
        element_references:
        num_batches: number of batches to use in fit. If not given will use all batches
        num_workers: number of workers to use in data loader
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
        num_workers=num_workers if num_workers is not None else batch_size,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )

    num_batches = num_batches if num_batches is not None else len(data_loader)
    if num_batches > len(data_loader):
        logging.warning(
            f"The give num_batches {num_batches} is larger than total batches of size {batch_size} in dataset. "
            f"Will ignore num_batches and use the whole dataset."
        )
        num_batches = len(data_loader)

    element_references = element_references or {}
    target_vectors = defaultdict(list)

    logging.info(
        f"Estimating mean and std for normalization using {num_batches * batch_size} samples in {num_batches} batches "
        f"of size {batch_size}."
    )
    for i, batch in tqdm(
        enumerate(data_loader), total=num_batches, desc="Estimating mean and std"
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
        normalizers[target] = create_normalizer(tensor=target_vector)

    return normalizers


def load_normalizers_from_config(
    config: dict[str, Any],
    dataset: Dataset,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
    element_references: dict[str, LinearReference] | None = None,
) -> dict[str, Normalizer]:
    """Create a dictionary with element references from a config."""
    return _load_from_config(
        config,
        "element_references",
        fit_normalizers,
        create_normalizer,
        dataset,
        checkpoint_dir,
        seed=seed,
        element_references=element_references,
    )
