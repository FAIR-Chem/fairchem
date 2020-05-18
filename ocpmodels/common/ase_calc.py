"""
Integrates models into an ASE calculator to be used for atomistic simulations
"""

from ase.calculators.calculator import Calculator


class OCP(Calculator):
    """OCP-ASE calculator"""

    implemented_properties = ["energy", "forces"]

    def __init__(self, trainer):
        Calculator.__init__(self)
        self.trainer = trainer
        # define model to be trained - trainer should contain all necessarry
        # information specified beforehand

    def train(self,):
        # TODO
        # train model
        self.trainer.train()

    def load(self, trained_model):
        # TODO
        # load previously trained model
        raise

    def calculate(self, atoms, properties, system_changes):
        # TODO predict energy and forces using trained model
        # Identify how atoms object is fed into model
        Calculator.calculate(self, atoms, properties, system_changes)

        predictions = self.trainer.predict(atoms)

        energy = None
        forces = None

        self.results["energy"] = energy
        self.results["forces"] = forces
