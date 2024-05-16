import os

import ase.io
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def v0_check(full_traj, initial):
    """
    Checks whether the initial structure as gathered from the POSCAR input file
    is in agreement with the initial image of the full trajectory. If not, the
    trajectory comes fro the V0 dataset which failed to save intermediate
    checkpoints.

    Args
    full_traj (list of Atoms objects): Calculated full trajectory.
    initial (Atoms object): Starting image provided by POSCAR..
    """
    initial_full = full_traj[0]
    error = np.mean(np.abs(initial_full.positions - initial.positions))
    return error


def restart_bug_check(full_traj):
    """
    Observed that some of the trajectories had a strange identically cyclical
    behavior - suggesting that a checkpoint was restarted from an earlier
    checkpoint rather than the latest. Checks whether the trajectory provided
    falls within that bug.

    Args
    full_traj (list of Atoms objects): Calculated full trajectory.
    """
    energy_set = set()
    for image in full_traj:
        energy = image.get_potential_energy(apply_constraint=False)
        if energy in energy_set:
            return True
            break
        else:
            energy_set.add(energy)
    return False


def plot_traj(traj, fname):
    """
    Plots the energy profile of a given trajectory

    Args
    traj (list of Atoms objects): Full trajectory to be plotted
    fname (str): Filename to be used as title and save figure as.
    """
    traj_energies = [
        image.get_potential_energy(apply_constraint=False) for image in traj
    ]
    steps = range(len(traj))
    plt.figure(figsize=(8, 8))
    plt.plot(steps, traj_energies)
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title(fname)
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/{fname}.png", dpi=300)
    plt.show()
