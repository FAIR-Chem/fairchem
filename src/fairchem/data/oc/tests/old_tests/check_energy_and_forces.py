import argparse
import multiprocessing as mp
import pickle

import numpy as np
from ase.io import Trajectory, read
from tqdm import tqdm


def check_relaxed_forces(sid, path, thres):
    """
    Check all forces in the final frame of adslab is less than a threshold.
    """
    final_atoms = read(path)
    forces = final_atoms.get_forces()
    if not (np.max(np.abs(forces)) <= thres):
        print(f"{sid} doesn't satisfy the force threshold, check trajectory {path}")


def check_adsorption_energy(sid, path, ref_energy, adsorption_energy):
    final_energy = read(path)
    if (
        not abs((final_energy.get_potential_energy() - ref_energy) - adsorption_energy)
        < 1e-6
    ):
        print(f"{sid} doesn't satify energy equation")


def check_DFT_energy(sid, path, e_tol=0.05):
    """
    Given a relaxation trajectory, check to see if 1. final energy is less than the initial
    energy, raise error if not. 2) If the energy decreases throuhghout a trajectory (small spikes are okay).
    And 3) if 2 fails, check if it's just a matter of tolerance being too strict by
    considering only the first quarter of the trajectory and sampling every 10th frame
    to check for _almost_ monotonic decrease in energies.
    If any frame(i+1) energy is higher than frame(i) energy, flag it and plot the trajectory.
    """
    traj = Trajectory(path)
    if traj[-1].get_potential_energy() > traj[0].get_potential_energy():
        print(
            "{} has final DFT energy that's higher than the initial energy, check traj {}".format(
                sid, path
            )
        )
    energies = [traj[i].get_potential_energy() for i in range(len(traj))]
    is_monotonic = all(
        energies[i + 1] - energies[i] < e_tol for i in range(len(energies) - 1)
    )
    if is_monotonic is False:
        print(
            "There is a spike in energy during the relaxation of {}, double check its trajectory {}".format(
                sid, path
            )
        )
        is_almost_monotonic = all(
            energies[i] >= energies[i + 10]
            for i in range(0, int(0.25 * len(energies)) - 10, 10)
        )
        if is_almost_monotonic is False:
            print(
                "almost_monotonic energy check fails, double check trajectory {}".format(
                    path
                )
            )


def check_positions_across_frames_are_different(sid, path):
    """
    Given a relaxation trajectory, make sure positions for two consecutive
    frames are not identical.
    """
    traj = Trajectory(path)
    positions = [traj[i].get_positions() for i in range(len(traj))]
    is_different = all(
        (positions[i] != positions[i + 1]).any() for i in range(len(positions) - 1)
    )
    if is_different is False:
        print(f"{sid} has identical positions for some frames, check {path}")


def read_pkl(fname):
    return pickle.load(open(fname, "rb"))


def run_checks(args):
    sysid_list, force_thres, traj_path_by_sysid, ref_energies, ads_energies = args
    for sysid in sysid_list:
        check_relaxed_forces(sysid, traj_path_by_sysid[sysid], force_thres)
        check_adsorption_energy(
            sysid, traj_path_by_sysid[sysid], ref_energies[sysid], ads_energies[sysid]
        )
        check_DFT_energy(sysid, traj_path_by_sysid[sysid])
        check_positions_across_frames_are_different(sysid, traj_path_by_sysid[sysid])


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sysid_file",
        type=str,
        help="A txt file constains all the system ids (str) of the dataset",
    )
    parser.add_argument(
        "--traj_path_by_sysid",
        type=str,
        help="A pickle file that contains a dictionary that maps trajectory path to system ids",
    )
    parser.add_argument(
        "--adsorption_energies",
        type=str,
        help="A pickle file that contains a dictionary that maps adsorption energy to system ids",
    )
    parser.add_argument(
        "--ref_energies",
        type=str,
        help="A pickle file that contains a dictionary that maps reference energy (E_slab + E_gas) to system ids",
    )
    parser.add_argument(
        "--force_tol",
        type=float,
        default=0.03,
        help="Force threshold at which a relaxation is considered converged",
    )
    parser.add_argument(
        "--e_tol",
        type=float,
        default=0.05,
        help="Energy threshold to flag a trajectory if potential energy of step i+1 is higher than step i by this amount",
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of processes or no. of dataset chunk"
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    sysids = open(args.sysid_file).read().splitlines()
    traj_path_by_sysid = read_pkl(args.traj_path_by_sysid)
    adsorption_energy_by_sysid = read_pkl(args.adsorption_energies)
    ref_energy_by_sysid = read_pkl(args.ref_energies)
    force_thres = args.force_tol
    mp_splits = np.array_split(sysids, args.num_workers)
    pool_args = [
        (
            split,
            force_thres,
            traj_path_by_sysid,
            ref_energy_by_sysid,
            adsorption_energy_by_sysid,
        )
        for split in mp_splits
    ]
    pool = mp.Pool(args.num_workers)
    tqdm(pool.imap(run_checks, pool_args), total=len(pool_args))
