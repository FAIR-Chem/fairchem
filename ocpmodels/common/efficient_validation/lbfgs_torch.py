from collections import deque

import torch
from ase import Atoms


class LBFGS:
    def __init__(self, atoms: Atoms, maxstep=0.04, memory=100, damping=1., alpha=70.,
                 force_consistent=None, num_steps=100, force_threshold=1e-8, device='cuda:0'):
        self.atoms = atoms
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        self.alpha = alpha
        self.force_consistent = force_consistent
        self.num_steps = num_steps
        self.force_threshold = force_threshold
        self.device = device

    def get_forces(self):
        return self.atoms.get_forces()

    def get_positions(self):
        return self.atoms.get_positions()

    def set_positions(self, update):
        r = self.get_positions()
        self.atoms.set_positions(r + update)

    def converged(self):
        forces = self.atoms.get_forces()
        return (forces ** 2).sum(axis=1).max() < self.force_threshold ** 2

    def run(self):
        s = deque(maxlen=self.memory)
        y = deque(maxlen=self.memory)
        rho = deque(maxlen=self.memory)
        r0 = f0 = None
        H0 = 1. / self.alpha

        with torch.no_grad():
            iteration = 0
            while iteration < self.num_steps and not self.converged():
                r0, f0 = self.step(iteration, r0, f0, H0, rho, s, y)
                iteration += 1
                forces = self.atoms.get_forces()
                print(iteration, np.sqrt((forces ** 2).sum(axis=1).max()))
                print(self.atoms.get_positions())

    def step(self, iteration, r0, f0, H0, rho, s, y):
        def determine_step(dr):
            steplengths = torch.norm(dr, dim=1)
            longest_step = steplengths.max()
            if longest_step >= self.maxstep:
                dr *= self.maxstep / longest_step
            return dr * self.damping

        f = torch.from_numpy(self.atoms.get_forces()).to(self.device)
        r = torch.from_numpy(self.atoms.get_positions()).to(self.device)

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
        self.atoms.set_positions((r + dr).cpu().numpy())
        return r, f


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
    opt = LBFGS(water, num_steps=100)
    opt.run()
