"""
Integrates models into an ASE calculator to be used for atomistic simulations
"""
import os

import ase.io
from ase.calculators.calculator import Calculator

from ocpmodels.common.registry import registry


class OCPCalculator(Calculator):
    """OCP-ASE calculator"""

    implemented_properties = ["energy", "forces"]

    def __init__(self, trainer):
        Calculator.__init__(self)
        self.trainer = trainer

    def train(self):
        self.trainer.train()

    def load(self, trained_model):
        # TODO
        # load previously trained model
        raise

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # TODO: allow atoms objects to be fed into model directly
        # rather than read from traj file to avoid this unnecessary I/O
        # TODO: do we want to hash the atoms object to store preprocessed data,
        # or discard entirely for each step. Storage issues likely with little
        # gain.
        ase.io.write("temp.traj", atoms)
        dataset_config = {"src": "./", "traj": "temp.traj"}
        predictions = self.trainer.predict(dataset_config, verbose=False)
        os.system("rm -rf processed/ temp.traj")

        self.results["energy"] = predictions["energy"][0]
        self.results["forces"] = predictions["forces"][0]
