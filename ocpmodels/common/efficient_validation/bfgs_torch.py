# -*- coding: utf-8 -*-
import warnings

import numpy as np
import torch
from torch_geometric.data import Batch

from ocpmodels.common.utils import radius_graph_pbc
from ocpmodels.preprocessing import AtomsToGraphs
from ase import Atoms


class BFGS:
    def __init__(self, atoms, model, maxstep=0.04, alpha=70):
        """BFGS optimizer.
        Parameters:
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
        self.model = model
        self.H = None
        self.H0 = (
            torch.eye(3 * len(self.atoms.atomic_numbers), dtype=torch.float64)
            * alpha
        ).cuda()  # TODO: add check for device and make it configurable
        self.r0 = None
        self.f0 = None
        self.nsteps = 0

    def converged(self, fmax):
        energy, forces = self.model.get_forces(self.atoms)
        print(self.nsteps, torch.sqrt((forces ** 2).sum(axis=1).max()))
        return (forces ** 2).sum(axis=1).max().item() < fmax ** 2

    def run(self, fmax=0.05, steps=100):
        while not self.converged(fmax) and self.nsteps < steps:

            self.step()
            self.nsteps += 1

            energy, forces = self.model.get_forces(self.atoms)
        self.atoms.y, self.atoms.force = self.model.get_forces(
            self.atoms, apply_constraint=False
        )
        return self.atoms

    def set_positions(self, update):
        r = self.atoms.pos
        self.atoms.pos = r + update.to(dtype=torch.float32)
        self.atoms = self.model.update_graph(self.atoms)

    def step(self):
        r = self.atoms.pos
        _, f = self.model.get_forces(self.atoms)
        f = f.reshape(-1)

        arg1 = r.view(-1).to(dtype=torch.float64)  # double()
        arg2 = f.to(dtype=torch.float64)

        self.update(arg1, arg2, self.r0, self.f0)

        omega1, V1 = torch.symeig(self.H, eigenvectors=True)

        dr1 = torch.matmul(V1, torch.matmul(arg2, V1) / torch.abs(omega1))
        dr1 = dr1.view(-1, 3)
        steplengths1 = (dr1 ** 2).sum(1) ** 0.5
        dr1 = self.determine_step(dr1, steplengths1)

        self.set_positions(dr1)

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
