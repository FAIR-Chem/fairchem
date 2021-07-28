import pickle
import os
import numpy as np
import sys
import multiprocessing as mp
import argparse
from ase.io import read, Trajectory
from tqdm import tqdm

def check_relaxed_forces(sid, path, thres):
    """
    Check all forces in the final frame of adslab is less than a threshold.
    """
    final_atoms = read(path)
    forces = final_atoms.get_forces()
    if not (np.max(np.abs(forces)) <= thres):
        print(f"{sid} doesn't satisfy the force threshold")

def check_adsorption_energy(sid, path, ref_energy, adsorption_energy):
    final_energy = read(path)
    if not abs((final_energy.get_potential_energy() - ref_energy) - adsorption_energy) < 1e-6:
        print(f"{sid} doesn't satify energy equation")

def check_DFT_energy(sid, path):
    """
    Given a relaxation trajectory, check to see if 1. final energy is less than the initial
    energy, raise error if not. 2) If energies monotonically decrease in a trajectory.
    If any frame(i+1) energy is slightly higher than frame(i) energy, flag it and plot the trajectory.
    """
    traj = Trajectory(path)
    if traj[-1].get_potential_energy() > traj[0].get_potential_energy():
        raise ValueError(f"{sid} has final DFT energy that's higher than the initial energy")
    flagged = False
    for idx, frame in enumerate(traj[:-1]):
        next_frame = traj[idx+1]
        diff_e = next_frame.get_potential_energy() - frame.get_potential_energy()
        if diff_e > 0.05:
            flagged = True
    if flagged:
        print('There is a spike in energy during the relaxation of {}, double check the trajectory'.format(sid))

def read_pkl(fname):
    return pickle.load(open(fname, 'rb'))

def run_checks(args):
    sysid_list, force_thres, traj_path_by_sysid, input_dir_by_sysid, ref_energies, ads_energies = args
    for sysid in sysid_list:
        check_relaxed_forces(sysid, traj_path_by_sysid[sysid], force_thres)
        check_adsorption_energy(sysid, traj_path_by_sysid[sysid],
                                ref_energies[sysid], ads_energies[sysid])
        check_DFT_energy(sysid, traj_path_by_sysid[sysid])

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sysid_file", type=str, help="A txt file constains all the system ids (str) of the dataset")
    parser.add_argument("--traj_path_by_sysid", type=str, help="A pickle file that contains a dictionary that maps trajectory path to system ids")
    parser.add_argument("--adsorption_energies", type=str, help="A pickle file that contains a dictionary that maps adsorption energy to system ids")
    parser.add_argument("--ref_energies", type=str, help="A pickle file that contains a dictionary that maps reference energy (E_slab + E_gas) to system ids")
    parser.add_argument("--force_tol", type=float, default=0.03, help="Force threshold at which a relaxation is considered converged")
    parser.add_argument("--num_workers",  type=int, help="Number of processes or no. of dataset chunk")
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
    pool_args = [(
                  split,
                  force_thres,
                  traj_path_by_sysid,
                  input_dir_by_sysid,
                  ref_energy_by_sysid,
                  adsorption_energy_by_sysid
    ) for split in mp_splits]
    pool = mp.Pool(args.num_workers)
    tqdm(pool.imap(run_checks, pool_args), total=len(pool_args))