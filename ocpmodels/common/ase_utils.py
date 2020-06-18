import os
from os import path

import ase.io
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize.optimize import Optimizer
from ase.optimize import BFGS

import numpy as np

from ocpmodels.common.registry import registry


class OCPCalculator(Calculator):
    implemented_properties = ["energy", "forces"]
    def __init__(self, trainer):
        """
        OCP-ASE Calculator

        Args:
            trainer: Object
                ML trainer for energy and force predictions.
        """
        Calculator.__init__(self)
        self.trainer = trainer

    def train(self):
        self.trainer.train()

    def load_pretrained(self, checkpoint_path):
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        # try:
            # self.trainer.load_pretrained(checkpoint_path)
        # except Exceptionk"Unable to load model!")

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # TODO: do we want to hash the atoms object to store preprocessed data,
        # or discard entirely for each step. Storage issues likely with little
        # gain.
        batch = self.trainer.dataset.ase_atoms_to_batch(atoms)
        predictions = self.trainer.predict(batch, verbose=False)

        self.results["energy"] = predictions["energy"][0]
        self.results["forces"] = predictions["forces"][0]


class Relaxation:
    def __init__(self, structure, filename, optimizer=BFGS):
        """
        ASE-based structure relaxations.

        Args:
            structure: Atoms object
                Atoms object to be relaxed

            filename: string
                Path to save generated trajectroy

            optimizer: Object
                Optimizer to be used for relaxations. Currently limited to
                ASE-based optimizers: https://wiki.fysik.dtu.dk/ase/ase/optimize.html.
        """
        assert isinstance(structure, Atoms), f"Invalid structure type! Expected {Atoms}"
        assert issubclass(optimizer, Optimizer), "Invalid optimizer!"
        self.structure = structure.copy()
        self.optimizer = optimizer
        self.filename = filename

    def run(self, calculator, steps=300, fmax=0.01, logfile=None):
        """Run structure relaxation

        Args:
            calculator: Object
                ASE-calculator to be used for energy/force predictions.

            steps: int
                Max number of steps in structure relaxation.

            fmax: float
                Structure relaxation terminates when the max force
                of the system is no bigger than fmax.

            logfile: path
                Logfile to store optimization stdout. '-' to print
                to terminal.
        """
        self.structure.set_calculator(calculator)
        dyn = self.optimizer(atoms=self.structure, trajectory=self.filename,
                logfile=logfile)
        dyn.run(fmax=fmax, steps=steps)

    def get_trajectory(self, full=False):
        """
        Retrieve generated trajectory

        Args:
            full: boolean
                True to return full trajectory, False to return only relaxed
                state"
        """
        assert path.exists(self.filename), "Trajectory not found!"
        full_trajectory = ase.io.read(self.filename, index=":")
        assert len(full_trajectory) > 0, "Trajectory empty!"

        return full_trajectory if full else full_trajectory[-1]

def relax_eval(trainer, filedir, steps=300, fmax=0.01):
    calc = OCPCalculator(trainer)

    mae_energy = 0
    mae_structure_ratio = 0
    os.makedirs("./ml_relax_trajs/", exist_ok=True)

    #TODO Parallelize ml-relaxations
    for traj in os.listdir(filedir):
        if traj.endswith("traj"):
            starting_image = ase.io.read(traj, "0")
            relaxed_image = ase.io.read(traj, "-1")
            structure_optimizer = Relaxation(starting_image, f"ml_relax_trajs/ml_relax_{traj}")
            structure_optimizer.run(calc, steps=steps, fmax=fmax)
            final_traj = structure_optimizer.get_trajectory()

            pred_final_energy = final_traj.get_potential_energy(apply_constraint=False)
            true_final_energy = relaxed_image.get_potential_energy(apply_constraint=False)
            energy_error = np.abs(true_final_energy - pred_final_energy)

            initial_structure_error = np.mean(
                    np.abs(starting_image.get_positions() - relaxed_image.get_positions())
                    )
            final_structure_error = np.mean(
                    np.abs(relaxed_image.get_positions() - final_traj.get_positions())
                    )
            structure_error_ratio = final_structure_error/initial_structure_error

            mae_energy += energy_error
            mae_structure_ratio += structure_error_ratio

    return mae_energy, mae_structure_ratio

