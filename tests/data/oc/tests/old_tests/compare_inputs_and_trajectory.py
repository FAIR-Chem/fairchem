import argparse
import multiprocessing as mp
import os
import pickle

import ase
import ase.io
import numpy as np
from tqdm import tqdm


def get_starting_structure_from_input_dir(input_dir):
    metadata_path = os.path.join(input_dir, "metadata.pkl")
    sort_path = os.path.join(input_dir, "ase-sort.dat")
    m = open(metadata_path, "rb")
    metadata = pickle.load(m)
    m.close()
    sorts = []
    with open(sort_path, "r") as f:
        for line in f:
            sort, resort = line.split()
            sorts.append(int(sort))
    # pre-vasp sort to post-vasp structure
    input_atoms = metadata["adsorbed_bulk_atomsobject"][sorts]
    return input_atoms


def min_diff(atoms_init, atoms_final):
    """
    Calculate atom wise distances of two atoms object,
    taking into account periodic boundary conditions.
    """
    positions = atoms_final.positions - atoms_init.positions
    fractional = np.linalg.solve(atoms_init.get_cell(complete=True).T, positions.T).T
    for i, periodic in enumerate(atoms_init.pbc):
        if periodic:
            fractional[:, i] %= 1.0
            fractional[:, i] %= 1.0
    fractional[fractional > 0.5] -= 1
    return np.matmul(fractional, atoms_init.get_cell(complete=True))


def compare(args):
    sysids, traj_path_by_sysid, input_dir_by_sysid = args
    for sysid in sysids:
        traj_path = traj_path_by_sysid[sysid]
        input_dir = input_dir_by_sysid[sysid]
        assert traj_path is not None
        assert input_dir is not None

        first_frame = ase.io.read(traj_path, index=0)
        ref_atoms = get_starting_structure_from_input_dir(input_dir)

        delta = min_diff(first_frame, ref_atoms)
        if not np.max(np.abs(delta)) < 1e-6:
            print(
                f"First frame of {sysid} trajectory doesn't match its atoms object input"
            )


def read_pkl(fname):
    return pickle.load(open(fname, "rb"))


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sysid_file",
        type=str,
        help="A txt file constains all the system ids (random ids) of the dataset",
    )
    parser.add_argument(
        "--traj_path_by_sysid",
        type=str,
        help="A pickle file that contains a dictionary that maps trajectory path to system ids",
    )
    parser.add_argument(
        "--input_dir_by_sysid",
        type=str,
        help="A pickle file that contains a dictionary that maps input folder path, which has metadata.pkl, to system ids",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of batch to split the inputs to preprocess",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    sysids = open(args.sysid_file).read().splitlines()
    traj_path_by_sysid = read_pkl(args.traj_path_by_sysid)
    input_dir_by_sysid = read_pkl(args.input_dir_by_sysid)

    sysids_splits = np.array_split(sysids, args.num_workers)
    pool_args = [
        (split, traj_path_by_sysid, input_dir_by_sysid) for split in sysids_splits
    ]
    pool = mp.Pool(args.num_workers)
    tqdm(pool.imap(compare, pool_args), total=len(pool_args))
