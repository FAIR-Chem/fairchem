"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.



Utilities to interface OCP models/trainers with the Atomic Simulation
Environment (ASE)
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.geometry import wrap_positions

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import (
    load_config,
    setup_imports,
    setup_logging,
    update_config,
)
from fairchem.core.datasets import data_list_collater
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.preprocessing import AtomsToGraphs

if TYPE_CHECKING:
    from torch_geometric.data import Batch


# system level model predictions have different shapes than expected by ASE
ASE_PROP_RESHAPE = MappingProxyType(
    {"stress": (-1, 3, 3), "dielectric_tensor": (-1, 3, 3)}
)


def batch_to_atoms(
    batch: Batch,
    results: dict[str, torch.Tensor] | None = None,
    wrap_pos: bool = True,
    eps: float = 1e-7,
) -> list[Atoms]:
    """Convert a data batch to ase Atoms

    Args:
        batch: data batch
        results: dictionary with predicted result tensors that will be added to a SinglePointCalculator. If no results
            are given no calculator will be added to the atoms objects.
        wrap_pos: wrap positions back into the cell.
        eps: Small number to prevent slightly negative coordinates from being wrapped.

    Returns:
        list of Atoms
    """
    n_systems = batch.natoms.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    fixed = torch.split(batch.fixed.to(torch.bool), natoms)
    if results is not None:
        results = {
            key: val.view(ASE_PROP_RESHAPE.get(key, -1)).tolist()
            if len(val) == len(batch)
            else [v.cpu().detach().numpy() for v in torch.split(val, natoms)]
            for key, val in results.items()
        }

    positions = torch.split(batch.pos, natoms)
    tags = torch.split(batch.tags, natoms)
    cells = batch.cell

    atoms_objects = []
    for idx in range(n_systems):
        pos = positions[idx].cpu().detach().numpy()
        cell = cells[idx].cpu().detach().numpy()

        # TODO take pbc from data
        if wrap_pos:
            pos = wrap_positions(pos, cell, pbc=[True, True, True], eps=eps)

        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            cell=cell,
            positions=pos,
            tags=tags[idx].tolist(),
            constraint=FixAtoms(mask=fixed[idx].tolist()),
            pbc=[True, True, True],
        )

        if results is not None:
            calc = SinglePointCalculator(
                atoms=atoms, **{key: val[idx] for key, val in results.items()}
            )
            atoms.set_calculator(calc)

        atoms_objects.append(atoms)

    return atoms_objects


class OCPCalculator(Calculator):
    """ASE based calculator using an OCP model"""

    _reshaped_props = ASE_PROP_RESHAPE

    def __init__(
        self,
        config_yml: str | None = None,
        checkpoint_path: str | Path | None = None,
        model_name: str | None = None,
        local_cache: str | None = None,
        trainer: str | None = None,
        cpu: bool = True,
        seed: int | None = None,
        only_output: list[str] | None = None,
        disable_amp: bool = True,
    ) -> None:
        """
        OCP-ASE Calculator

        Args:
            config_yml (str):
                Path to yaml config or could be a dictionary.
            checkpoint_path (str):
                Path to trained checkpoint.
            model_name (str):
                Model name to use. Pretrained model checkpoint will be
                downloaded if not found in your local_cache.
            local_cache (str):
                Directory to save pretrained model checkpoints.
            trainer (str):
                OCP trainer to be used. "forces" for S2EF, "energy" for IS2RE.
            cpu (bool):
                Whether to load and run the model on CPU. Set `False` for GPU.
            seed (int):
                Seed in the calculator to ensure reproducibility among instantiations
            only_output (list):
                A list of outputs to use from the model, rather than relying on the underlying task
            disable_amp (bool):
                Disable AMP in the calculator; AMP on is great for training, but often leads to headaches
                during inference.
        """
        setup_imports()
        setup_logging()
        super().__init__()

        if model_name is not None:
            if checkpoint_path is not None:
                raise RuntimeError(
                    "model_name and checkpoint_path were both specified, please use only one at a time"
                )
            if local_cache is None:
                raise NotImplementedError(
                    "Local cache must be set when specifying a model name"
                )
            checkpoint_path = model_name_to_local_file(
                model_name=model_name, local_cache=local_cache
            )

        checkpoint_path = Path(checkpoint_path)

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or checkpoint_path is not None

        checkpoint = None
        if config_yml is not None:
            if isinstance(config_yml, str):
                config, duplicates_warning, duplicates_error = load_config(config_yml)
                if len(duplicates_warning) > 0:
                    logging.warning(
                        f"Overwritten config parameters from included configs "
                        f"(non-included parameters take precedence): {duplicates_warning}"
                    )
                if len(duplicates_error) > 0:
                    raise ValueError(
                        f"Conflicting (duplicate) parameters in simultaneously "
                        f"included configs: {duplicates_error}"
                    )
            else:
                config = config_yml

            # Only keeps the train data that might have normalizer values
            if isinstance(config["dataset"], list):
                config["dataset"] = config["dataset"][0]
            elif isinstance(config["dataset"], dict):
                config["dataset"] = config["dataset"].get("train", None)
        else:
            # Loads the config from the checkpoint directly (always on CPU).
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            config = checkpoint["config"]

        if trainer is not None:
            config["trainer"] = trainer
        else:
            config["trainer"] = config.get("trainer", "ocp")

        if "model_attributes" in config:
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        # Calculate the edge indices on the fly
        config["model"]["otf_graph"] = True

        ### backwards compatability with OCP v<2.0
        config = update_config(config)

        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = str(checkpoint_path)
        del config["dataset"]["src"]

        # some models that are published have configs that include tasks
        # which are not output by the model
        if only_output is not None:
            assert isinstance(
                only_output, list
            ), "only output must be a list of targets to output"
            for key in only_output:
                assert (
                    key in config["outputs"]
                ), f"{key} listed in only_outputs is not present in current model outputs {config['outputs'].keys()}"
            remove_outputs = set(config["outputs"].keys()) - set(only_output)
            for key in remove_outputs:
                config["outputs"].pop(key)

        self.trainer = registry.get_trainer_class(config["trainer"])(
            task=config.get("task", {}),
            model=config["model"],
            dataset=[config["dataset"]],
            outputs=config["outputs"],
            loss_functions=config["loss_functions"],
            evaluation_metrics=config["evaluation_metrics"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=cpu,
            amp=config.get("amp", False),
            inference_only=True,
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path=checkpoint_path, checkpoint=checkpoint)

        seed = seed if seed is not None else self.trainer.config["cmd"]["seed"]
        if seed is None:
            logging.warning(
                "No seed has been set in modelcheckpoint or OCPCalculator! Results may not be reproducible on re-run"
            )
        else:
            self.trainer.set_seed(seed)

        if disable_amp:
            self.trainer.scaler = None

        self.a2g = AtomsToGraphs(
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_pbc=True,
            r_edges=not self.trainer.model.otf_graph,  # otf graph should not be a property of the model...
        )
        self.implemented_properties = list(self.config["outputs"].keys())

    def load_checkpoint(
        self, checkpoint_path: str, checkpoint: dict | None = None
    ) -> None:
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
            checkpoint: dict
                A pretrained checkpoint dict
        """
        try:
            self.trainer.load_checkpoint(
                checkpoint_path, checkpoint, inference_only=True
            )
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def calculate(self, atoms: Atoms | Batch, properties, system_changes) -> None:
        """Calculate implemented properties for a single Atoms object or a Batch of them."""
        super().calculate(atoms, properties, system_changes)
        if isinstance(atoms, Atoms):
            data_object = self.a2g.convert(atoms)
            batch = data_list_collater([data_object], otf_graph=True)
        else:
            batch = atoms

        predictions = self.trainer.predict(batch, per_image=False, disable_tqdm=True)

        for key in predictions:
            _pred = predictions[key]
            _pred = _pred.item() if _pred.numel() == 1 else _pred.cpu().numpy()
            if key in OCPCalculator._reshaped_props:
                _pred = _pred.reshape(OCPCalculator._reshaped_props.get(key)).squeeze()
            self.results[key] = _pred
