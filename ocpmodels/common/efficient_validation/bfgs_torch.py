# -*- coding: utf-8 -*-
import warnings

import numpy as np
import torch
from torch_geometric.data import Batch

from ocpmodels.preprocessing import AtomsToGraphs


class BFGS:
    def __init__(self, atoms, maxstep=0.04, alpha=70):
        """BFGS optimizer.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.04 Å).

        """
        if maxstep > 1.0:
            warnings.warn(
                "You are using a much too large value for "
                "the maximum step size: %.1f Å" % maxstep
            )
        self.maxstep = maxstep
        self.atoms = atoms
        self.H = None
        self.H0 = torch.eye(3 * len(self.atoms), dtype=torch.float64) * alpha
        self.r0 = None
        self.f0 = None
        self.nsteps = 0

    def converged(self, fmax):
        forces = self.atoms.get_forces()
        return (forces ** 2).sum(axis=1).max() < fmax ** 2

    def run(self, fmax=0.05, steps=100):
        while not self.converged(fmax) and self.nsteps < steps:

            self.step()
            self.nsteps += 1

            forces = self.atoms.get_forces()
            print(self.nsteps, np.sqrt((forces ** 2).sum(axis=1).max()))

    def get_forces(self):
        return self.atoms.get_forces()

    def get_positions(self):
        return self.atoms.get_positions()

    def set_positions(self, update):
        r = self.get_positions()
        self.atoms.set_positions(r + update)

    def step(self):
        r = self.get_positions()
        f = self.get_forces()
        f = f.reshape(-1)

        arg1 = torch.tensor(r, dtype=torch.float64).view(-1)
        arg2 = torch.tensor(
            f, dtype=torch.float64
        )  # f has to be converted to torch tensor

        self.update(arg1, arg2, self.r0, self.f0)

        omega1, V1 = torch.symeig(self.H, eigenvectors=True)
        dr1 = torch.matmul(V1, torch.matmul(arg2, V1) / torch.abs(omega1))
        dr1 = dr1.view(-1, 3)
        steplengths1 = (dr1 ** 2).sum(1) ** 0.5
        dr1 = self.determine_step(dr1, steplengths1)

        self.set_positions(dr1.numpy())

        self.r0 = arg1.clone()
        self.f0 = arg2.clone()

    def determine_step(self, dr, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = torch.max(steplengths)
        if maxsteplength >= self.maxstep:
            dr *= self.maxstep / maxsteplength

        return dr

    def update(self, r, f, r0, f0):
        if self.H is None:
            self.H = self.H0
            return
        dr = r - r0

        if torch.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        df = f - f0
        a = torch.dot(dr, df)
        dg = torch.matmul(self.H, dr)
        b = torch.dot(dr, dg)
        val = (
            torch.ger(df, df) / a + torch.ger(dg, dg) / b
        )  # To batchify: https://discuss.pytorch.org/t/batch-outer-product/4025/4
        self.H -= val


class TorchCalc:
    def __init__(self, atoms, trainer):
        self.atoms = atoms
        self.trainer = trainer
        # TODO: torchify AtomsToGraphs preprocessing call
        self.a2g = AtomsToGraphs(
            max_neigh=12,
            radius=6,
            r_energy=False,
            r_forces=False,
            r_distances=False,
        )
        self.data_object = self.get_data_object(self.atoms)

    def __len__(self):
        return len(self.atoms)

    def get_data_object(self, atoms):
        return self.a2g.convert(atoms)

    def get_forces(self):
        self.batch = Batch.from_data_list([self.data_object])
        predictions = self.trainer.predict(self.batch)
        return predictions["forces"][0]

    def get_positions(self):
        return self.data_object.pos

    def set_positions(self, update):
        self.data_object.pos = update
        self.update_graph()
        return self.data_object.pos

    def update_graph(self):
        self.atoms.set_positions(self.data_object.pos.numpy())
        self.data_object = self.get_data_object(self.atoms)
