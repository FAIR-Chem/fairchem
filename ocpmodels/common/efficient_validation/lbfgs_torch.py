from collections import deque

import torch
from ase import Atoms

from ocpmodels.common.utils import radius_graph_pbc


class LBFGS:
    def __init__(self, atoms: Atoms, model, maxstep=0.04, memory=50, damping=1., alpha=70.,
                 force_consistent=None, device='cuda:0'):
        self.atoms = atoms
        self.model = model
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        self.alpha = alpha
        self.force_consistent = force_consistent
        self.device = device

    def get_forces(self):
        energy, forces = self.model.get_forces(self.atoms)
        return energy, forces

    def get_positions(self):
        return self.atoms.pos

    def set_positions(self, update):
        r = self.get_positions()
        self.atoms.pos = r + update.to(dtype=torch.float32)
        self.model.update_graph(self.atoms)

    def converged(self, force_threshold):
        _, forces = self.get_forces()
        return (forces ** 2).sum(axis=1).max() < force_threshold ** 2

    def run(self, fmax, steps):
        s = deque(maxlen=self.memory)
        y = deque(maxlen=self.memory)
        rho = deque(maxlen=self.memory)
        r0 = f0 = None
        H0 = 1. / self.alpha

        # with torch.no_grad():
        iteration = 0
        while iteration < steps and not self.converged(fmax):
            r0, f0 = self.step(iteration, r0, f0, H0, rho, s, y)
            iteration += 1
            energy, forces = self.get_forces()
            print(iteration, torch.sqrt((forces ** 2).sum(axis=1).max()).item())
        self.atoms.force = forces
        self.atoms.y = energy
        return self.atoms

    def step(self, iteration, r0, f0, H0, rho, s, y):
        def determine_step(dr):
            steplengths = torch.norm(dr, dim=1)
            longest_step = steplengths.max()
            if longest_step >= self.maxstep:
                dr *= self.maxstep / longest_step
            return dr * self.damping

        # f = torch.from_numpy(self.get_forces()).to(self.device, dtype=torch.float64)
        f = self.get_forces()[1].to(self.device, dtype=torch.float64)
        r = self.atoms.pos.to(self.device, dtype=torch.float64)

        # Update s, y and rho
        if iteration > 0:
            s0 = (r - r0).flatten()
            y0 = -(f - f0).flatten()
            s.append(s0)
            y.append(y0)
            rho.append(1. / torch.dot(y0, s0))

        loopmax = min(self.memory, iteration)
        alpha = f.new_empty(loopmax)
        q = -f.flatten()

        for i in range(loopmax - 1, -1, -1):
            alpha[i] = rho[i] * torch.dot(s[i], q)
            q -= alpha[i] * y[i]
        z = H0 * q
        for i in range(loopmax):
            beta = rho[i] * torch.dot(y[i], z)
            z += s[i] * (alpha[i] - beta)
        p = - z.reshape((-1, 3))  # descent direction
        dr = determine_step(p)
        if torch.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return
        self.set_positions(dr)
        return r, f


class TorchCalc:
    def __init__(self, model):
        self.model = model

    def get_forces(self, atoms):
        predictions = self.model.predict(atoms)
        return predictions["energy"], predictions["forces"]

    def update_graph(self, atoms):
        edge_index, cell_offsets, num_neighbors = radius_graph_pbc(
            atoms, 6, 200, atoms.pos.device
        )
        atoms.edge_index = edge_index
        atoms.cell_offsets = cell_offsets
        atoms.neighbors = num_neighbors
        return atoms


if __name__ == '__main__':
    from ase import Atoms

    from ase.calculators.emt import EMT
    import numpy as np

    d = 0.9575
    t = np.pi / 180 * 104.51
    water = Atoms('H2O',
                  positions=[(d, 0, 0),
                             (d * np.cos(t), d * np.sin(t), 0),
                             (0, 0, 0)],
                  calculator=EMT())
    opt = LBFGS(water)
    opt.run(num_steps=100)
