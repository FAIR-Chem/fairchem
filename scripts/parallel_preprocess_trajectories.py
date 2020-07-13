"""
Reads a folder of Trajectory files and saves an LMDB with extracted features.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import random

import ase.io
import lmdb
import numpy as np
import torch
from ase.io.trajectory import Trajectory
from tqdm import tqdm

from ocpmodels.preprocessing import AtomsToGraphs


def read_trajectory_and_extract_features(a2g, traj_path):
    traj = ase.io.read(traj_path, ":")
    return a2g.convert_all(traj, disable_tqdm=True)


def construct_lmdb(inputs):
    idx, paths = inputs
    os.makedirs(os.path.join(args.out_path, f"db_{idx}"), exist_ok=True)
    map_size = 1099511627776 * 2
    db = lmdb.open(
        os.path.join(args.out_path, f"db_{idx}", f"data_{idx}.lmdb"),
        map_size=map_size,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Extract features.
    i = 0
    for path in paths:
        try:
            dl = read_trajectory_and_extract_features(a2g, path)
            for do in dl:
                # filter out images with excessivley large forces, if applicable
                if torch.max(torch.abs(do.force)).item() <= args.filter:
                    txn = db.begin(write=True)
                    txn.put(
                        f"{i}".encode("ascii"), pickle.dumps(do, protocol=-1)
                    )
                    txn.commit()
                    i += 1
        except Exception:
            continue

    db.sync()
    db.close()


if __name__ == "__main__":
    # Parse a few arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj-paths-txt",
        default=None,
        required=True,
        help="Path to txt file containing trajectory paths (one per line)",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        required=True,
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--filter",
        type=float,
        default=1.0e9,
        help="Max force to filter out, default: no filter",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=50,
        help="Number of cpus to parallelize across",
    )

    args = parser.parse_args()

    # Read the txt file with trajectory paths.
    with open(os.path.join(args.traj_paths_txt), "r") as f:
        raw_traj_files = f.read().splitlines()
    num_trajectories = len(raw_traj_files)

    print(
        "### Found %d trajectories in %s"
        % (num_trajectories, args.traj_paths_txt)
    )

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=12,
        radius=6,
        dummy_distance=7,
        dummy_index=-1,
        r_energy=True,
        r_forces=True,
        r_distances=False,
    )

    raw_traj_splits = np.array_split(np.array(raw_traj_files), args.cpus)
    _inputs = []
    for idx, i in enumerate(range(1, args.cpus + 1)):
        _inputs.append((i, raw_traj_splits[idx]))

    pool = mp.Pool(args.cpus)
    list(tqdm(pool.imap(construct_lmdb, _inputs)))
