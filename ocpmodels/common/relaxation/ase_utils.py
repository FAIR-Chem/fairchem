"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.



Utilities to interface OCP models/trainers with the Atomic Simulation
Environment (ASE)
"""
import copy
import logging
import os

import torch
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.constraints import FixAtoms

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    radius_graph_pbc,
    setup_imports,
    setup_logging,
)
from ocpmodels.datasets import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs


def batch_to_atoms(batch):
    n_systems = batch.neighbors.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    fixed = torch.split(batch.fixed, natoms)
    forces = torch.split(batch.force, natoms)
    positions = torch.split(batch.pos, natoms)
    tags = torch.split(batch.tags, natoms)
    cells = batch.cell
    energies = batch.y.tolist()

    atoms_objects = []
    for idx in range(n_systems):
        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=positions[idx].cpu().detach().numpy(),
            tags=tags[idx].tolist(),
            cell=cells[idx].cpu().detach().numpy(),
            constraint=FixAtoms(mask=fixed[idx].tolist()),
            pbc=[True, True, True],
        )
        calc = sp(
            atoms=atoms,
            energy=energies[idx],
            forces=forces[idx].cpu().detach().numpy(),
        )
        atoms.set_calculator(calc)
        atoms_objects.append(atoms)

    return atoms_objects


class OCPCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        config_yml=None,
        checkpoint=None,
        trainer=None,
        cutoff=6,
        max_neighbors=50,
        cpu=True,
    ):
        """
        OCP-ASE Calculator

        Args:
            config_yml (str):
                Path to yaml config or could be a dictionary.
            checkpoint (str):
                Path to trained checkpoint.
            trainer (str):
                OCP trainer to be used. "forces" for S2EF, "energy" for IS2RE.
            cutoff (int):
                Cutoff radius to be used for data preprocessing.
            max_neighbors (int):
                Maximum amount of neighbors to store for a given atom.
            cpu (bool):
                Whether to load and run the model on CPU. Set `False` for GPU.
        """
        setup_imports()
        setup_logging()
        Calculator.__init__(self)

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or checkpoint is not None

        if config_yml is not None:
            if isinstance(config_yml, str):
                config = yaml.safe_load(open(config_yml, "r"))

                if "includes" in config:
                    for include in config["includes"]:
                        # Change the path based on absolute path of config_yml
                        path = os.path.join(
                            config_yml.split("configs")[0], include
                        )
                        include_config = yaml.safe_load(open(path, "r"))
                        config.update(include_config)
            else:
                config = config_yml
            # Only keeps the train data that might have normalizer values
            if isinstance(config["dataset"], list):
                config["dataset"] = config["dataset"][0]
            elif isinstance(config["dataset"], dict):
                config["dataset"] = config["dataset"].get("train", None)
        else:
            # Loads the config from the checkpoint directly (always on CPU).
            config = torch.load(checkpoint, map_location=torch.device("cpu"))[
                "config"
            ]
        if trainer is not None:  # passing the arg overrides everything else
            config["trainer"] = trainer
        else:
            if "trainer" not in config:  # older checkpoint
                if config["task"]["dataset"] == "trajectory_lmdb":
                    config["trainer"] = "forces"
                elif config["task"]["dataset"] == "single_point_lmdb":
                    config["trainer"] = "energy"
                else:
                    logging.warning(
                        "Unable to identify OCP trainer, defaulting to `forces`. Specify the `trainer` argument into OCPCalculator if otherwise."
                    )
                    config["trainer"] = "forces"

        if "model_attributes" in config:
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        # for checkpoints with relaxation datasets defined, remove to avoid
        # unnecesarily trying to load that dataset
        if "relax_dataset" in config["task"]:
            del config["task"]["relax_dataset"]

        # Calculate the edge indices on the fly
        config["model"]["otf_graph"] = True

        # Save config so obj can be transported over network (pkl)
        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = checkpoint

        if "normalizer" not in config:
            del config["dataset"]["src"]
            config["normalizer"] = config["dataset"]

        self.trainer = registry.get_trainer_class(
            config.get("trainer", "energy")
        )(
            task=config["task"],
            model=config["model"],
            dataset=None,
            normalizer=config["normalizer"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=cpu,
        )

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        self.a2g = AtomsToGraphs(
            max_neigh=max_neighbors,
            radius=cutoff,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
            r_pbc=True,
        )

    def load_checkpoint(self, checkpoint_path):
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        data_object = self.a2g.convert(atoms)
        batch = data_list_collater([data_object], otf_graph=True)

        predictions = self.trainer.predict(
            batch, per_image=False, disable_tqdm=True
        )
        if self.trainer.name == "s2ef":
            self.results["energy"] = predictions["energy"].item()
            self.results["forces"] = predictions["forces"].cpu().numpy()

        elif self.trainer.name == "is2re":
            self.results["energy"] = predictions["energy"].item()
