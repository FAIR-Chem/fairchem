"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import ase
import torch
from torch_scatter import scatter

from fairchem.core.common.relaxation.ase_utils import batch_to_atoms
from fairchem.core.common.relaxation.optimizers.optimize import OptimizableBatch


class LBFGS:
    def __init__(
        self,
        optimizable_batch: OptimizableBatch,
        maxstep: float = 0.01,
        memory: int = 100,
        damping: float = 0.25,
        alpha: float = 100.0,
        force_consistent=None,
        device: str = "cuda:0",
        save_full_traj: bool = True,
        traj_dir: Path | None = None,
        traj_names=None,
        early_stop_batch: bool = False,
    ) -> None:
        self.optimizable = optimizable_batch
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        self.alpha = alpha
        self.H0 = 1.0 / self.alpha
        self.force_consistent = force_consistent
        self.device = device
        self.save_full = save_full_traj
        self.traj_dir = traj_dir
        self.traj_names = traj_names
        self.early_stop_batch = early_stop_batch
        self.otf_graph = optimizable_batch.trainer._unwrapped_model.otf_graph

        assert not self.traj_dir or (
            traj_dir and len(traj_names)
        ), "Trajectory names should be specified to save trajectories"
        logging.info("Step   Fmax(eV/A)")

        if not self.otf_graph and "edge_index" not in self.optimizable.batch:
            self.optimizable.update_graph()

    def set_positions(self, update, update_mask):
        if not self.early_stop_batch:
            update = torch.where(update_mask.unsqueeze(1), update, 0.0)
        self.optimizable.batch.pos += update.to(dtype=torch.float32)

        if not self.otf_graph:
            self.optimizable.update_graph()

    def check_convergence(self, iteration, forces=None, energy=None):
        if energy is None:
            energy = self.optimizable.get_potential_energies()

        if forces is None:
            forces = self.optimizable.get_forces(apply_constraint=True)
            forces = forces.to(dtype=torch.float64)

        max_forces_ = scatter(
            (forces**2).sum(axis=1).sqrt(), self.optimizable.batch.batch, reduce="max"
        )
        logging.info(
            f"{iteration} " + " ".join(f"{x:0.3f}" for x in max_forces_.tolist())
        )

        # (batch_size) -> (nAtoms)
        max_forces = max_forces_[self.optimizable.batch.batch]

        return max_forces.lt(self.fmax), energy, forces

    def run(self, fmax, steps):
        self.fmax = fmax
        self.steps = steps

        self.s = deque(maxlen=self.memory)
        self.y = deque(maxlen=self.memory)
        self.rho = deque(maxlen=self.memory)
        self.r0 = self.f0 = None

        self.trajectories = None
        if self.traj_dir:
            self.traj_dir.mkdir(exist_ok=True, parents=True)
            self.trajectories = [
                ase.io.Trajectory(self.traj_dir / f"{name}.traj_tmp", mode="w")
                for name in self.traj_names
            ]

        iteration = 0
        converged = False
        converged_mask = torch.zeros_like(
            self.optimizable.batch.atomic_numbers, device=self.device
        ).bool()
        while iteration < steps and not converged:
            _converged_mask, energy, forces = self.check_convergence(iteration)
            # Models like GemNet-OC can have random noise in their predictions.
            # Here we ensure atom positions are not being updated after already
            # hitting the desired convergence criteria.
            converged_mask = torch.logical_or(converged_mask, _converged_mask)
            converged = torch.all(converged_mask)
            update_mask = torch.logical_not(converged_mask)

            if self.trajectories is not None and (
                self.save_full or converged or iteration == steps - 1 or iteration == 0
            ):
                self.write(energy, forces, update_mask)

            if not converged and iteration < steps - 1:
                self.step(iteration, forces, update_mask)

            iteration += 1

        # GPU memory usage as per nvidia-smi seems to gradually build up as
        # batches are processed. This releases unoccupied cached memory.
        torch.cuda.empty_cache()

        if self.trajectories is not None:
            for traj in self.trajectories:
                traj.close()
            for name in self.traj_names:
                traj_fl = Path(self.traj_dir / f"{name}.traj_tmp", mode="w")
                traj_fl.rename(traj_fl.with_suffix(".traj"))

        self.optimizable.batch.y = self.optimizable.get_potential_energies()
        self.optimizable.batch.force = self.optimizable.get_forces(
            apply_constraint=False
        )

        return self.optimizable.batch

    def step(
        self,
        iteration: int,
        forces: torch.Tensor | None,
        update_mask: torch.Tensor,
    ) -> None:
        def _batched_dot(x: torch.Tensor, y: torch.Tensor):
            return scatter(
                (x * y).sum(dim=-1), self.optimizable.batch.batch, reduce="sum"
            )

        def determine_step(dr):
            steplengths = torch.norm(dr, dim=1)
            longest_steps = scatter(
                steplengths, self.optimizable.batch.batch, reduce="max"
            )
            longest_steps = longest_steps[self.optimizable.batch.batch]
            maxstep = longest_steps.new_tensor(self.maxstep)
            scale = (longest_steps + 1e-7).reciprocal() * torch.min(
                longest_steps, maxstep
            )
            dr *= scale.unsqueeze(1)
            return dr * self.damping

        if forces is None:
            forces = self.optimizable.get_forces(apply_constraint=True)

        r = self.optimizable.batch.pos.clone().to(dtype=torch.float64)

        # Update s, y, rho
        if iteration > 0:
            s0 = r - self.r0
            self.s.append(s0)

            y0 = -(forces - self.f0)
            self.y.append(y0)

            self.rho.append(1.0 / _batched_dot(y0, s0))

        loopmax = min(self.memory, iteration)
        alpha = forces.new_empty(loopmax, self.optimizable.batch.natoms.shape[0])
        q = -forces

        for i in range(loopmax - 1, -1, -1):
            alpha[i] = self.rho[i] * _batched_dot(self.s[i], q)  # b
            q -= alpha[i][self.optimizable.batch.batch, ..., None] * self.y[i]

        z = self.H0 * q
        for i in range(loopmax):
            beta = self.rho[i] * _batched_dot(self.y[i], z)
            z += self.s[i] * (
                alpha[i][self.optimizable.batch.batch, ..., None]
                - beta[self.optimizable.batch.batch, ..., None]
            )

        # descent direction
        p = -z
        dr = determine_step(p)
        if torch.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        self.set_positions(dr, update_mask)

        self.r0 = r
        self.f0 = forces

    def write(self, energy, forces, update_mask) -> None:
        self.optimizable.batch.y, self.optimizable.batch.force = energy, forces
        atoms_objects = batch_to_atoms(self.optimizable.batch)
        update_mask_ = torch.split(update_mask, self.optimizable.batch.natoms.tolist())
        for atm, traj, mask in zip(atoms_objects, self.trajectories, update_mask_):
            if mask[0] or not self.save_full:
                traj.write(atm)
