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

    def load_pretrained(self, checkpoint_path):
        self.trainer.load_pretrained(checkpoint_path)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # TODO: do we want to hash the atoms object to store preprocessed data,
        # or discard entirely for each step. Storage issues likely with little
        # gain.
        batch = self.trainer.dataset.ase_atoms_to_batch(atoms)
        predictions = self.trainer.predict(batch, verbose=False)

        self.results["energy"] = predictions["energy"][0]
        self.results["forces"] = predictions["forces"][0]
