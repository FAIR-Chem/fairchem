"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
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
    from torch_geometric.data import Batch


class LinearReferences(nn.Module):
    """Represents an elemental linear references model for a target property.

    In an elemental reference associates a value with each chemical element present in the dataset.
    Elemental references define a chemical composition model, i.e. a rough approximation of a target
    property (energy) using elemental references is done by summing the elemental references multiplied
    by the number of times the corresponding element is present.

    Elemental references energies can be taken as:
     - the energy of a chemical species in its elemental state
       (i.e. lowest energy polymorph of single element crystal structures for solids)
     - fitting a linear model to a dataset, where the features are the counts of each element in each data point.
       see the function fit_linear references below for details

    Training GNNs to predict the difference between DFT and the predictions of a chemical composition
    model represent a useful normalization scheme that can improve model accuracy. See for example the
    "Alternative reference scheme" section of the OC22 manuscript: https://arxiv.org/pdf/2206.08917
    """

    def __init__(
        self,
        element_references: torch.Tensor | None = None,
        max_num_elements: int = 118,
    ):
        """
        Args:
            element_references (Tensor): tensor with linear reference values
            max_num_elements (int): max number of elements - 118 is a stretch
            metrics (dict): dictionary with accuracy metrics in predicting values for structures used in fitting.
        """
        super().__init__()
        self.register_buffer(
            name="element_references",
            tensor=element_references
            if element_references is not None
            else torch.zeros(max_num_elements + 1),
        )

    def _apply_refs(
        self, target: torch.Tensor, batch: Batch, sign: int, reshaped: bool = True
    ) -> torch.Tensor:
        """Apply references batch-wise"""
        indices = batch.atomic_numbers.to(
            dtype=torch.int, device=self.element_references.device
        )
        elemrefs = self.element_references[indices].to(dtype=target.dtype)
        # this option should not exist, all tensors should have compatible shapes in dataset and trainer outputs
        if reshaped:
            elemrefs = elemrefs.view(batch.natoms.sum(), -1)

        return target.index_add(0, batch.batch, elemrefs, alpha=sign)

    @torch.autocast(device_type="cuda", enabled=False)
    def dereference(
        self, target: torch.Tensor, batch: Batch, reshaped: bool = True
    ) -> torch.Tensor:
        """Remove linear references"""
        return self._apply_refs(target, batch, -1, reshaped=reshaped)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        self, target: torch.Tensor, batch: Batch, reshaped: bool = True
    ) -> torch.Tensor:
        """Add linear references"""
        return self._apply_refs(target, batch, 1, reshaped=reshaped)


def create_element_references(
    file: str | Path | None = None,
    state_dict: dict | None = None,
) -> LinearReferences:
    """Create an element reference module.

    Args:
        type (str): type of reference (only linear implemented)
        file (str or Path): path to pt or npz file
        state_dict (dict): a state dict of a element reference module

    Returns:
        LinearReference
    """
    if file is not None and state_dict is not None:
        logging.warning(
            "Both a file and a state_dict for element references was given."
            "The references will be read from the file and the provided state_dict will be ignored."
        )

    # path takes priority if given
    if file is not None:
        extension = Path(file).suffix
        if extension == ".pt":
            # try to load a pt file
            state_dict = torch.load(file)
        elif extension == ".npz":
            state_dict = {}
            with np.load(file) as values:
                # legacy linref files
                if "coeff" in values:
                    state_dict["element_references"] = torch.tensor(values["coeff"])
                else:
                    state_dict["element_references"] = torch.tensor(
                        values["element_references"]
                    )
        else:
            raise RuntimeError(
                f"Element references file with extension '{extension}' is not supported."
            )

    if "element_references" not in state_dict:
        raise RuntimeError("Unable to load linear element references!")

    return LinearReferences(element_references=state_dict["element_references"])


@torch.no_grad()
def fit_linear_references(
    targets: list[str],
    dataset: Dataset,
    batch_size: int,
    num_batches: int | None = None,
    num_workers: int = 0,
    max_num_elements: int = 118,
    log_metrics: bool = True,
    use_numpy: bool = True,
    driver: str | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> dict[str, LinearReferences]:
    """Fit a set linear references for a list of targets using a given number of batches.

    Args:
        targets: list of target names
        dataset: data set to fit linear references with
        batch_size: size of batch
        num_batches: number of batches to use in fit. If not given will use all batches
        num_workers: number of workers to use in data loader.
            Note setting num_workers > 1 leads to finicky multiprocessing issues when using this function
            in distributed mode. The issue has to do with pickling the functions in load_references_from_config
            see function below...
        max_num_elements: max number of elements in dataset. If not given will use an ambitious value of 118
        log_metrics: if true will compute MAE, RMSE and R2 score of fit and log.
        use_numpy: use numpy.linalg.lstsq instead of torch. This tends to give better solutions.
        driver: backend used to solve linear system. See torch.linalg.lstsq docs. Ignored if use_numpy=True
        shuffle: whether to shuffle when loading the dataset
        seed: random seed used to shuffle the sampler if shuffle=True

    Returns:
        dict of fitted LinearReferences objects
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

    max_num_elements += 1  # + 1 since H starts at index 1
    # solving linear system happens on CPU, which allows handling poorly conditioned and
    # rank deficient matrices, unlike torch lstsq on GPU
    composition_matrix = torch.zeros(
        num_batches * batch_size,
        max_num_elements,
    )

    target_vectors = {
        target: torch.zeros(num_batches * batch_size) for target in targets
    }

    logging.info(
        f"Fitting linear references using {num_batches * batch_size} samples in {num_batches} "
        f"batches of size {batch_size}."
    )
    for i, batch in tqdm(
        enumerate(data_loader), total=num_batches, desc="Fitting linear references"
    ):
        if i == 0:
            assert all(
                len(batch[target].squeeze().shape) == 1 for target in targets
            ), "element references can only be used for scalar targets"
        elif i == num_batches:
            break

        next_batch_size = len(batch) if i == len(data_loader) - 1 else batch_size
        for target in targets:
            target_vectors[target][
                i * batch_size : i * batch_size + next_batch_size
            ] = batch[target].to(torch.float64)
        for j, data in enumerate(batch.to_data_list()):
            composition_matrix[i * batch_size + j] = torch.bincount(
                data.atomic_numbers.int(),
                minlength=max_num_elements,
            ).to(torch.float64)

    # reduce the composition matrix to only features that are non-zero to improve rank
    mask = composition_matrix.sum(axis=0) != 0.0
    reduced_composition_matrix = composition_matrix[:, mask]
    elementrefs = {}

    for target in targets:
        coeffs = torch.zeros(max_num_elements)

        if use_numpy:
            solution = torch.tensor(
                np.linalg.lstsq(
                    reduced_composition_matrix.numpy(),
                    target_vectors[target].numpy(),
                    rcond=None,
                )[0]
            )
        else:
            lstsq = torch.linalg.lstsq(
                reduced_composition_matrix, target_vectors[target], driver=driver
            )
            solution = lstsq.solution

        coeffs[mask] = solution
        elementrefs[target] = LinearReferences(coeffs)

        if log_metrics is True:
            y = target_vectors[target]
            y_pred = torch.matmul(reduced_composition_matrix, solution)
            y_mean = target_vectors[target].mean()
            N = len(target_vectors[target])
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - y_mean) ** 2).sum()
            mae = (abs(y - y_pred)).sum() / N
            rmse = (((y - y_pred) ** 2).sum() / N).sqrt()
            r2 = 1 - (ss_res / ss_tot)
            logging.info(
                f"Training accuracy metrics for fitted linear element references: mae={mae}, rmse={rmse}, r2 score={r2}"
            )

    return elementrefs


def load_references_from_config(
    config: dict[str, Any],
    dataset: Dataset,
    seed: int = 0,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, LinearReferences]:
    """Create a dictionary with element references from a config."""
    return _load_from_config(
        config,
        "element_references",
        fit_linear_references,
        create_element_references,
        dataset,
        checkpoint_dir,
        seed=seed,
    )
