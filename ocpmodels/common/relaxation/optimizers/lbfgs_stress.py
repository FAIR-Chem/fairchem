import logging
from collections import deque
from pathlib import Path

import ase
import torch
from ase import Atoms
from torch_scatter import scatter

from ocpmodels.common.relaxation.ase_utils import batch_to_atoms
from ocpmodels.common.utils import radius_graph_pbc



class LBFGS_Stress:
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
        self.cur_deform_grad = None

    def get_forces(self, apply_constraint=True):
        energy, forces = self.model.get_forces(self.atoms, apply_constraint)
        return energy, forces

    def get_positions(self):
        return self.atoms.pos

    
    def get_cell(self):
        return self.atoms.cell

    def set_positions_cell(self, pos, update, update_mask):        
        if not self.early_stop_batch:
            update = torch.where(update_mask.unsqueeze(1), update, 0.0)
        new_pos = (pos + update).to(dtype=torch.float32)
        nAtoms = len(self.atoms.batch)
        new_pos_atom = new_pos[:nAtoms, :]
        new_deform_grad = new_pos[nAtoms:, :].reshape(-1, 3, 3)
        self.atoms.cell = torch.bmm(self.get_cell(), torch.transpose(new_deform_grad, 1, 2))
        self.atoms.pos = torch.bmm(new_pos_atom.reshape(-1, 1, 3), torch.transpose(new_deform_grad[self.atoms.batch, :, :].reshape(-1, 3, 3), 1, 2)).reshape(-1, 3)
        self.model.update_graph(self.atoms)

    def check_convergence(
        self, iteration, update_mask, forces, force_threshold
    ):
        if forces is None:
            return False
        
        list_batch = []
        for i in range(len(self.atoms.natoms)):
            list_batch += [i] * 3
        torch_batch = torch.cat([self.atoms.batch, torch.Tensor(list_batch).to(self.atoms.batch.device)], dim=0).to(self.atoms.batch.dtype)

        max_forces_ = scatter(
            (forces**2).sum(axis=1).sqrt(), torch_batch, reduce="max"
        )
        max_forces = max_forces_[torch_batch]
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
        list_batch = []
        for i in range(len(self.atoms.natoms)):
            list_batch += [i] * 3
        torch_batch = torch.cat([self.atoms.batch, torch.Tensor(list_batch).to(self.atoms.batch.device)], dim=0).to(self.atoms.batch.dtype)

        update_mask = torch.ones_like(torch_batch).bool().to(self.device)

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
                nAtoms = len(self.atoms.batch)
                self.atoms.y, self.atoms.force = e0, f0[:nAtoms, :]
                atoms_objects = batch_to_atoms(self.atoms)
                update_mask_ = torch.split(
                    update_mask, (self.atoms.natoms + 3).tolist() # it is garbage, have to modify ittt
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
        energy, augmented_forces = self.get_forces(
            apply_constraint=False
        )
        self.atoms.y, self.atoms.force = energy, augmented_forces[:nAtoms, :]
        return self.atoms

    def step(self, iteration, r0, f0, H0, rho, s, y, update_mask):
        def determine_step(dr):
            steplengths = torch.norm(dr, dim=1)
            list_batch = []
            for i in range(len(self.atoms.natoms)):
                list_batch += [i] * 3
            torch_batch = torch.cat([self.atoms.batch, torch.Tensor(list_batch).to(self.atoms.batch.device)], dim=0).to(self.atoms.batch.dtype)
            longest_steps = scatter(
                steplengths, torch_batch, reduce="max"
            )
            longest_steps = longest_steps[torch_batch]
            maxstep = longest_steps.new_tensor(self.maxstep)
            scale = (longest_steps + 1e-7).reciprocal() * torch.min(
                longest_steps, maxstep
            )
            dr *= scale.unsqueeze(1)
            return dr * self.damping
        e, f = self.get_forces()
        f = f.to(self.device, dtype=torch.float64)
        cell_eye = torch.stack([torch.eye(3).reshape(1, 3, 3) for _ in range(len(e))], dim=0).to(self.device)
        r = torch.cat([self.get_positions(), cell_eye.reshape(-1, 3)]).to(self.device, dtype=torch.float64)
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
        self.set_positions_cell(r, dr, update_mask)
        return r, f, e




class TorchCalcStress:
    def __init__(self, model, isotropic=False, opt_forces=True, opt_stress=True, cell_factor=0.1, transform=None):
        self.model = model
        self.transform = transform
        self.opt_forces = opt_forces
        self.opt_stress = opt_stress
        self.cell_factor = cell_factor
        self.isotropic = isotropic

    def get_forces(self, atoms, exp=False, apply_constraint=False):
            
        predictions = self.model.predict(
            atoms, per_image=False, disable_tqdm=True
        )
        energy = predictions["energy"] # (nMolecules, 1)
        forces = predictions["forces"] # (nAtoms, 3)
        stress = predictions["stress"].reshape(-1, 3, 3) # (nMolecules, 3, 3)
        scaled_stress = stress.reshape(-1, 3, 3) # (nMolecules, 3, 3) maybe add scale
        cell = atoms.cell # (nMolecules, 3, 3)
        volume = torch.sum(
                    cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                    dim=1,
                    keepdim=True,
                )[:, :, None] # (nMolecules, 1)
        scaled_virial = volume.reshape(-1, 1, 1) * scaled_stress * self.cell_factor # (nMolecules, 3, 3) check the units and the implementation
        if self.opt_forces:
            deformed_forces = forces
        else:
            deformed_forces = torch.zeros(forces.shape).to(device=forces.device, dtype=forces.dtype)
        
        if self.opt_stress:
            if self.isotropic: 
                trace = scaled_virial[:, 0, 0] + scaled_virial[:, 1, 1] + scaled_virial[:, 2, 2]
                scaled_virial = trace.reshape(-1, 1, 1)
                scaled_virial[:, 0, 1] = scaled_virial[:, 0, 2] = scaled_virial[:, 1, 0] = scaled_virial[:, 1, 2] = scaled_virial[:, 2, 0] = scaled_virial[:, 2, 1] = 0
                scaled_virial[:, 0, 0] = scaled_virial[:, 1, 1] = scaled_virial[:, 2, 2] = trace / 3
            scaled_virial = scaled_virial.reshape(-1, 3)
        else:
            scaled_virial = torch.zeros(scaled_virial.reshape(-1, 3).shape).to(device=scaled_virial.device, dtype=scaled_virial.dtype)
        augmented_forces = torch.concat([deformed_forces, scaled_virial.reshape(-1, 3)], dim=0) # (nAtoms + nMolecules * 3, 3)
        return energy, augmented_forces
    


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