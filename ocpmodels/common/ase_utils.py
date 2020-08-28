import os
from os import path

import ase.io
import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from ase.optimize.optimize import Optimizer

from ocpmodels.common.meter import mae, mae_ratio, mean_l2_distance
from ocpmodels.common.registry import registry
from ocpmodels.datasets.trajectory_lmdb import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs


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
        self.a2g = AtomsToGraphs(
            max_neigh=200,
            radius=6,
            r_energy=False,
            r_forces=False,
            r_distances=False,
        )

    def train(self):
        self.trainer.train()

    def load_pretrained(self, checkpoint_path):
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            self.trainer.load_pretrained(checkpoint_path)
        except NotImplementedError:
            print("Unable to load checkpoint!")

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        data_object = self.a2g.convert(atoms)
        batch = data_list_collater([data_object])
        predictions = self.trainer.predict(batch)

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
        assert isinstance(
            structure, Atoms
        ), f"Invalid structure type! Expected {Atoms}"
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
        dyn = self.optimizer(
            atoms=self.structure, trajectory=self.filename, logfile=logfile
        )
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


def relax_eval(trainer, traj_dir, metric, steps, fmax, results_dir):
    """
    Evaluation of ML-based relaxations.

    Args:

        trainer: object
            Trainer class necessary to build OCP-ASE calculator

        traj_dir: str
            Directory path containing trajectory files to be run
            by the model.

        metric: str
            Evaluation metric to be used.

        steps: int
            Max number of steps in the structure relaxation.

        fmax: float
                Structure relaxation terminates when the max force
                of the system is no bigger than fmax.

        results_dir: str
            Path to save model generated relaxations.

    """
    calc = OCPCalculator(trainer)

    mae_energy = []
    mae_structure = []
    os.makedirs(os.path.join(results_dir, "ml_relaxations"), exist_ok=True)

    # TODO Parallelize ml-relaxations
    for traj in os.listdir(traj_dir):
        if traj.endswith("traj"):
            traj_path = os.path.join(traj_dir, traj)
            initial_structure = ase.io.read(traj_path, "0")
            dft_relaxed_structure = ase.io.read(traj_path, "-1")

            # Run ML-based relaxation
            structure_optimizer = Relaxation(
                initial_structure, f"{results_dir}/ml_relaxations/ml_{traj}"
            )
            structure_optimizer.run(calc, steps=steps, fmax=fmax)
            ml_trajectory = structure_optimizer.get_trajectory(full=True)
            ml_relaxed_structure = ml_trajectory[-1]

            # Compute relaxed energy MAE
            ml_final_energy = torch.tensor(
                ml_relaxed_structure.get_potential_energy(
                    apply_constraint=False
                )
            )
            dft_final_energy = torch.tensor(
                dft_relaxed_structure.get_potential_energy(
                    apply_constraint=False
                )
            )
            energy_error = eval(metric)(dft_final_energy, ml_final_energy)

            # Compute relaxed structure MAE
            dft_relaxed_structure_positions = torch.tensor(
                dft_relaxed_structure.get_positions()
            )
            ml_relaxed_structure_positions = torch.tensor(
                ml_relaxed_structure.get_positions()
            )
            # TODO Explore alternative structure metrics
            structure_error = torch.mean(
                eval(metric)(
                    ml_relaxed_structure_positions,
                    dft_relaxed_structure_positions,
                )
            )

            mae_energy.append(energy_error)
            mae_structure.append(structure_error)

    # Average across all test systems
    mae_energy = torch.mean(torch.tensor(mae_energy))
    mae_structure = torch.mean(torch.tensor(mae_structure))

    return mae_energy, mae_structure
