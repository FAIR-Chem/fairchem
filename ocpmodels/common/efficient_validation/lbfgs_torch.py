from collections import deque

import torch
from ase import Atoms
from ase.constraints import FixAtoms

from ocpmodels.common.utils import radius_graph_pbc
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets.trajectory_lmdb import data_list_collater


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

    def converged(self, force_threshold, iteration):
        _, forces = self.get_forces()
        free_idx = torch.where(self.atoms.fixed == 0)[0]
        free_forces = forces[free_idx]
        print(iteration, torch.sqrt((free_forces ** 2).sum(axis=1).max()).item())
        return (free_forces ** 2).sum(axis=1).max() < force_threshold ** 2

    def run(self, fmax, steps):
        s = deque(maxlen=self.memory)
        y = deque(maxlen=self.memory)
        rho = deque(maxlen=self.memory)
        r0 = f0 = None
        H0 = 1. / self.alpha

        # with torch.no_grad():
        iteration = 0
        while iteration < steps and not self.converged(fmax, iteration):
            r0, f0 = self.step(iteration, r0, f0, H0, rho, s, y)
            iteration += 1
            energy, forces = self.get_forces()
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
    def __init__(self, model, pbc_graph=False):
        self.model = model
        self.pbc_graph = pbc_graph
        self.a2g = AtomsToGraphs(
            max_neigh=50,
            radius=6,
            r_energy=False,
            r_forces=False,
            r_distances=False,
        )

    def get_forces(self, atoms):
        predictions = self.model.predict(atoms)
        return predictions["energy"], predictions["forces"]

    def update_graph(self, atoms):
        if self.pbc_graph:
            edge_index, cell_offsets, num_neighbors = radius_graph_pbc(
                atoms, 6, 50, atoms.pos.device
            )
            atoms.edge_index = edge_index
            atoms.cell_offsets = cell_offsets
            atoms.neighbors = num_neighbors
        else:
            atoms_object = data_to_atoms(atoms)
            data_object = self.a2g.convert(atoms_object)
            atoms = data_list_collater([data_object])
        return atoms


def data_to_atoms(data):
    atoms = Atoms(
        numbers=data.atomic_numbers.tolist(),
        cell=data.cell.view(3, 3).cpu().detach().numpy(),
        positions=data.pos.cpu().detach().numpy(),
        constraint=FixAtoms(mask=data.fixed.tolist())
    )
    return atoms
