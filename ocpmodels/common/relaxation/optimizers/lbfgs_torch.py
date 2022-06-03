"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from collections import deque
from pathlib import Path

import ase
import torch
from ase import Atoms
from torch_scatter import scatter

from ocpmodels.common.relaxation.ase_utils import batch_to_atoms
from ocpmodels.common.utils import radius_graph_pbc


class LBFGS:
    def __init__(
        self,
        atoms: Atoms,
        model,
        maxstep=0.01,
        memory=100,
        damping=0.25,
        alpha=100.0,
        force_consistent=None,
        device="cuda:0",
        traj_dir: Path = None,
        traj_names=None,
        early_stop_batch: bool = False,
    ):
        self.atoms = atoms
        self.model = model
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        self.alpha = alpha
        self.force_consistent = force_consistent
        self.device = device
        self.traj_dir = traj_dir
        self.traj_names = traj_names
        self.early_stop_batch = early_stop_batch
        assert not self.traj_dir or (
            traj_dir and len(traj_names)
        ), "Trajectory names should be specified to save trajectories"
        logging.info("Step   Fmax(eV/A)")

        self.model.update_graph(self.atoms)

    def get_forces(self, apply_constraint=True):
        energy, forces = self.model.get_forces(self.atoms, apply_constraint)
        return energy, forces

    def get_positions(self):
        return self.atoms.pos

    def set_positions(self, update, update_mask):
        r = self.get_positions()
        if not self.early_stop_batch:
            update = torch.where(update_mask.unsqueeze(1), update, 0.0)
        self.atoms.pos = r + update.to(dtype=torch.float32)
        self.model.update_graph(self.atoms)

    def check_convergence(
        self, iteration, update_mask, forces, force_threshold
    ):
        if forces is None:
            return False
        max_forces_ = scatter(
            (forces**2).sum(axis=1).sqrt(), self.atoms.batch, reduce="max"
        )
        max_forces = max_forces_[self.atoms.batch]
        update_mask = torch.logical_and(
            update_mask, max_forces.ge(force_threshold)
        )
        logging.info(
            f"{iteration} "
            + " ".join(f"{x:0.3f}" for x in max_forces_.tolist())
        )
        return update_mask

    def run(self, fmax, steps):
        s = deque(maxlen=self.memory)
        y = deque(maxlen=self.memory)
        rho = deque(maxlen=self.memory)
        r0 = f0 = e0 = None
        H0 = 1.0 / self.alpha
        update_mask = torch.ones_like(self.atoms.batch).bool().to(self.device)

        trajectories = None
        if self.traj_dir:
            self.traj_dir.mkdir(exist_ok=True, parents=True)
            trajectories = [
                ase.io.Trajectory(self.traj_dir / f"{name}.traj_tmp", mode="w")
                for name in self.traj_names
            ]

        iteration = 0
        converged = False
        while iteration < steps and not converged:
            r0, f0, e0 = self.step(
                iteration, r0, f0, H0, rho, s, y, update_mask
            )
            iteration += 1
            if trajectories is not None:
                self.atoms.y, self.atoms.force = e0, f0
                atoms_objects = batch_to_atoms(self.atoms)
                update_mask_ = torch.split(
                    update_mask, self.atoms.natoms.tolist()
                )
                for atm, traj, mask in zip(
                    atoms_objects, trajectories, update_mask_
                ):
                    if mask[0]:
                        traj.write(atm)
            update_mask = self.check_convergence(
                iteration, update_mask, f0, fmax
            )
            converged = torch.all(torch.logical_not(update_mask))
        # GPU memory usage as per nvidia-smi seems to gradually build up as
        # batches are processed. This releases unoccupied cached memory.
        torch.cuda.empty_cache()

        if trajectories is not None:
            for traj in trajectories:
                traj.close()
            for name in self.traj_names:
                traj_fl = Path(self.traj_dir / f"{name}.traj_tmp", mode="w")
                traj_fl.rename(traj_fl.with_suffix(".traj"))

        self.atoms.y, self.atoms.force = self.get_forces(
            apply_constraint=False
        )
        return self.atoms

    def step(self, iteration, r0, f0, H0, rho, s, y, update_mask):
        def determine_step(dr):
            steplengths = torch.norm(dr, dim=1)
            longest_steps = scatter(
                steplengths, self.atoms.batch, reduce="max"
            )
            longest_steps = longest_steps[self.atoms.batch]
            maxstep = longest_steps.new_tensor(self.maxstep)
            scale = (longest_steps + 1e-7).reciprocal() * torch.min(
                longest_steps, maxstep
            )
            dr *= scale.unsqueeze(1)
            return dr * self.damping

        e, f = self.get_forces()
        f = f.to(self.device, dtype=torch.float64)
        r = self.atoms.pos.to(self.device, dtype=torch.float64)

        # Update s, y and rho
        if iteration > 0:
            s0 = (r - r0).flatten()
            y0 = -(f - f0).flatten()
            s.append(s0)
            y.append(y0)
            rho.append(1.0 / torch.dot(y0, s0))

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
        p = -z.reshape((-1, 3))  # descent direction
        dr = determine_step(p)
        if torch.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return
        self.set_positions(dr, update_mask)
        return r, f, e


class TorchCalc:
    def __init__(self, model, transform=None):
        self.model = model
        self.transform = transform

    def get_forces(self, atoms, apply_constraint=True):
        predictions = self.model.predict(
            atoms, per_image=False, disable_tqdm=True
        )
        energy = predictions["energy"]
        forces = predictions["forces"]
        if apply_constraint:
            fixed_idx = torch.where(atoms.fixed == 1)[0]
            forces[fixed_idx] = 0
        return energy, forces

    def update_graph(self, atoms):
        edge_index, cell_offsets, num_neighbors = radius_graph_pbc(
            atoms, 6, 50
        )
        atoms.edge_index = edge_index
        atoms.cell_offsets = cell_offsets
        atoms.neighbors = num_neighbors
        if self.transform is not None:
            atoms = self.transform(atoms)
        return atoms
