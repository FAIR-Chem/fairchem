import logging
from collections import deque
from pathlib import Path
from typing import Optional

import ase
import torch
from ase import Atoms
from torch_scatter import scatter

from ocpmodels.common.relaxation.ase_utils import batch_to_atoms
from ocpmodels.common.utils import radius_graph_pbc
from scipy import linalg


def logm(M):
    L, V = torch.linalg.eig(M)
    return torch.matmul(torch.matmul(V, torch.diag_embed(torch.log(L))), torch.linalg.inv(V)).real

def expm(M):
    L, V = torch.linalg.eig(M)
    return torch.matmul(torch.matmul(V, torch.diag_embed(torch.exp(L))), torch.linalg.inv(V)).real

def logm_scipy(M):
    B, S, _ = M.shape
    M_np = M.cpu().numpy()
    M_list = [torch.Tensor(linalg.logm(M_np[i, :, :])).reshape(1, S, S) for i in range(B)]
    M_log = torch.cat(M_list, dim=0).to(M.device)
    return M_log

def expm_scipy(M):
    B, S, _ = M.shape
    M_np = M.cpu().numpy()
    M_list = [torch.Tensor(linalg.expm(M_np[i, :, :])).reshape(1, S, S) for i in range(B)]
    M_exp = torch.cat(M_list, dim=0).to(M.device)
    return M_exp

class LBFGS_StressExp:
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
        save_full_traj=True,
        traj_dir: Path = None,
        traj_names=None,
        early_stop_batch: bool = False,
        scipy: bool = False,
    ):
        if scipy:
            self.expm = expm_scipy
            self.logm = logm_scipy
        else:
            self.expm = torch.matrix_exp
            self.logm = logm

        self.atoms = atoms
        self.orig_cell = self.atoms.cell.to(device)
        self.model = model
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
        self.otf_graph = model.model._unwrapped_model.otf_graph

        assert not self.traj_dir or (
            traj_dir and len(traj_names)
        ), "Trajectory names should be specified to save trajectories"
        logging.info("Step   Fmax(eV/A)")
        if not self.otf_graph and "edge_index" not in atoms:
            self.model.update_graph(self.atoms)

    def get_forces(self, apply_constraint=True):
        energy, atoms_forces, stress = self.model.get_forces(self.atoms, apply_constraint)
        cell = self.atoms.cell # (nMolecules, 3, 3)
        volume = torch.sum(
                    cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                    dim=1,
                    keepdim=True,
                )[:, :, None] # (nMolecules, 1)

        cur_deform_grad = self.deform_grad()

        #cur_deform_grad_np = cur_deform_grad.reshape(-1, 3, 3).cpu().numpy()
        #B = cur_deform_grad_np.shape[0]

        #cur_deform_grad_np_list = [torch.Tensor(linalg.logm(cur_deform_grad_np[i, :, :])).reshape(1, 3, 3) for i in range(B)]
        #cur_deform_grad_np_log = torch.cat(cur_deform_grad_np_list, dim=0).to(cur_deform_grad.device)
        #cur_deform_grad_log = cur_deform_grad_np_log
        cur_deform_grad_log = self.logm(cur_deform_grad.reshape(-1, 3, 3))

        atoms_forces = torch.bmm(atoms_forces.reshape(-1, 1, 3), cur_deform_grad[self.atoms.batch]).reshape(-1, 3)

        virial = volume.reshape(-1, 1, 1) * stress  # (nMolecules, 3, 3) check the units and the implementation
        virial = -torch.transpose(torch.linalg.solve(cur_deform_grad, torch.transpose(virial, 1, 2)), 1, 2)
        B = virial.shape[0]

        Y = torch.zeros((B, 6, 6), device = virial.device)
        Y[:, 0:3, 0:3] = cur_deform_grad_log
        Y[:, 3:6, 3:6] = cur_deform_grad_log
        Y[:, 0:3, 3:6] = - torch.bmm(virial, self.expm(-cur_deform_grad_log))
        deform_grad_log_force = -self.expm(Y)[:, 0:3, 3:6]
        for (i1, i2) in [(0, 1), (0, 2), (1, 2)]:
            ff = 0.5 * (deform_grad_log_force[:, i1, i2] +
                        deform_grad_log_force[:, i2, i1])
            deform_grad_log_force[:, i1, i2] = ff
            deform_grad_log_force[:, i2, i1] = ff

        # check for reasonable alignment between naive and
        # exact search directions
        all_are_equal = torch.all(torch.isclose(deform_grad_log_force,
                                          virial))
        torch.sum(deform_grad_log_force * virial) / torch.sqrt(torch.sum(deform_grad_log_force**2) * torch.sum(virial**2)) > 0.8
        if all_are_equal or \
            (torch.sum(deform_grad_log_force * virial) /
             torch.sqrt(torch.sum(deform_grad_log_force**2) *
                     torch.sum(virial**2)) > 0.8):
            deform_grad_log_force = virial
            
        augmented_forces = torch.concat([atoms_forces.reshape(-1, 3), deform_grad_log_force.reshape(-1, 3)], dim=0) # (nAtoms + nMolecules * 3, 3)
        return energy, augmented_forces

    def get_positions(self):
        cur_deform_grad = self.deform_grad()
        pos_atoms = torch.linalg.solve(cur_deform_grad[self.atoms.batch, :, :],
                                       self.atoms.pos.reshape(-1, 3, 1)).reshape(-1, 3)   # TODO: check if it is correct!
        
        #cur_deform_grad_np = cur_deform_grad.reshape(-1, 3, 3).cpu().numpy()
        #B = cur_deform_grad_np.shape[0]

        #cur_deform_grad_np_list = [torch.Tensor(linalg.logm(cur_deform_grad_np[i, :, :])).reshape(1, 3, 3) for i in range(B)]
        #cur_deform_grad_log = torch.cat(cur_deform_grad_np_list, dim=0).to(cur_deform_grad.device)
        #return torch.cat([pos_atoms, cur_deform_grad_log.reshape(-1, 3)]).to(self.device, dtype=torch.float64)
        return torch.cat([pos_atoms, self.logm(cur_deform_grad.reshape(-1, 3, 3)).reshape(-1, 3)]).to(self.device, dtype=torch.float64)

    def get_cell(self):
        return self.atoms.cell
    
    def deform_grad(self):
        return torch.transpose(torch.linalg.solve(self.orig_cell, self.atoms.cell), 1, 2)

    def set_positions(self, pos, update, update_mask):        
        if not self.early_stop_batch:
            update = torch.where(update_mask.unsqueeze(1), update, 0.0)
        new_pos = (pos + update).to(dtype=torch.float32)
        nAtoms = len(self.atoms.batch)
        new_pos_atom = new_pos[:nAtoms, :]
        new_deform_grad_log = new_pos[nAtoms:, :].reshape(-1, 3, 3)
        new_deform_grad = self.expm(new_deform_grad_log)
        self.atoms.cell = torch.bmm(self.orig_cell, torch.transpose(new_deform_grad, 1, 2))
        self.atoms.pos = torch.bmm(new_pos_atom.reshape(-1, 1, 3), torch.transpose(new_deform_grad[self.atoms.batch, :, :].reshape(-1, 3, 3), 1, 2)).reshape(-1, 3)
        if not self.otf_graph:
            self.model.update_graph(self.atoms)


    def check_convergence(self, iteration, forces=None, energy=None):
        if forces is None or energy is None:
            energy, forces = self.get_forces(self.atoms)
            forces = forces.to(dtype=torch.float64)
        max_forces_ = scatter(
            (forces**2).sum(axis=1).sqrt(), self.batch_aug, reduce="max"
        )
        logging.info(
            f"{iteration} "
            + " ".join(f"{x:0.3f}" for x in max_forces_.tolist())
        )

        # (batch_size) -> (nAtoms)
        max_forces = max_forces_[self.batch_aug]

        return max_forces.ge(self.fmax), energy, forces


    def run(self, fmax, steps):
        list_batch = []
        for i in range(len(self.atoms.natoms)):
            list_batch += [i] * 3
        self.batch_aug = torch.cat([self.atoms.batch.to(self.device), torch.Tensor(list_batch).to(self.device)], dim=0).to(self.device, self.atoms.batch.dtype)

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
        while iteration < steps and not converged:
            update_mask, energy, forces = self.check_convergence(iteration)
            update_mask = torch.ones_like(self.batch_aug).bool().to(self.device)
            converged = torch.all(torch.logical_not(update_mask))
            if self.trajectories is not None:
                if (
                    self.save_full
                    or converged
                    or iteration == steps - 1
                    or iteration == 0
                ):
                    self.write(energy, forces[:len(self.atoms.batch)], update_mask)

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

        energy, forces, stress = self.model.get_forces(
            self.atoms,
            apply_constraint=False,
        )
        self.atoms.y, self.atoms.force = energy.clone(), forces.clone()
        return self.atoms


    def step(
        self,
        iteration: int,
        forces: Optional[torch.Tensor],
        update_mask: torch.Tensor,
    ):
        def _batched_dot(x: torch.Tensor, y: torch.Tensor):
            return scatter((x * y).sum(dim=-1), self.batch_aug, reduce="sum")

        def determine_step(dr):
            steplengths = torch.norm(dr, dim=1)
            longest_steps = scatter(
                steplengths, self.batch_aug, reduce="max"
            )
            longest_steps = longest_steps[self.batch_aug]
            maxstep = longest_steps.new_tensor(self.maxstep)
            scale = (longest_steps + 1e-7).reciprocal() * torch.min(
                longest_steps, maxstep
            )
            dr *= scale.unsqueeze(1)
            return dr * self.damping

        if forces is None:
            _, forces = self.get_forces()

        r = self.get_positions().clone().to(dtype=torch.float64)

        # Update s, y, rho
        if iteration > 0:
            s0 = r - self.r0
            self.s.append(s0)

            y0 = -(forces - self.f0)
            self.y.append(y0)

            self.rho.append(1.0 / _batched_dot(y0, s0))

        loopmax = min(self.memory, iteration)
        alpha = forces.new_empty(loopmax, self.atoms.natoms.shape[0])
        q = -forces
        for i in range(loopmax - 1, -1, -1):
            alpha[i] = self.rho[i] * _batched_dot(self.s[i], q)  # b
            q -= alpha[i][self.batch_aug, ..., None] * self.y[i]

        z = self.H0 * q
        for i in range(loopmax):
            beta = self.rho[i] * _batched_dot(self.y[i], z)
            z += self.s[i] * (
                alpha[i][self.batch_aug, ..., None]
                - beta[self.batch_aug, ..., None]
            )

        # descent direction
        p = -z
        dr = determine_step(p)
        if torch.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return
        self.set_positions(r, dr, update_mask)

        self.r0 = r
        self.f0 = forces



    def write(self, energy, forces, update_mask):
        self.atoms.y, self.atoms.force = energy.clone(), forces.clone()
        atoms_objects = batch_to_atoms(self.atoms)
        update_mask_ = [True for _ in range(len(self.atoms.natoms))] # torch.split(update_mask, self.atoms.natoms.tolist())
        for atm, traj in zip(
            atoms_objects, self.trajectories
        ):
            #if mask[0] or not self.save_full:
            traj.write(atm)




class TorchCalcStressExp:
    def __init__(self, model, force_weight=1, stress_weight=0.1, transform=None):
        self.model = model
        self.transform = transform
        self.force_weight = force_weight
        self.stress_weight = stress_weight

    def get_forces(self, atoms, apply_constraint=False):
        predictions = self.model.predict(
            atoms, per_image=False, disable_tqdm=True
        )
        energy = predictions["energy"] # (nMolecules, 1)
        forces = predictions["forces"] # (nAtoms, 3)
        stress = predictions["stress"].reshape(-1, 3, 3) # (nMolecules, 3, 3)
        scaled_stress = stress.reshape(-1, 3, 3) * self.stress_weight # (nMolecules, 3, 3) maybe add scale
        if apply_constraint:
            fixed_idx = torch.where(atoms.fixed == 1)[0]
            forces[fixed_idx] = 0
        return energy, forces * self.force_weight, -scaled_stress
    


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
