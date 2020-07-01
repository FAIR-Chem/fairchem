"""
Reads a folder of Trajectory files and saves an LMDB with extracted features.
"""

import argparse
import os
import pickle

import lmdb
from ase.io.trajectory import Trajectory
from tqdm import tqdm

from ocpmodels.preprocessing import AtomsToGraphs


def read_trajectory_and_extract_features(a2g, traj_path):
    traj = Trajectory(traj_path)
    return a2g.convert_all(traj, disable_tqdm=True)


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
        help="Folder to save extracted features. Will create if doesn't exist",
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

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb.
    map_size = 1099511627776 * 2
    db = lmdb.open(
        os.path.join(args.out_path, "data.lmdb"),
        map_size=map_size,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Extract features.
    idx = 0
    for i in tqdm(range(num_trajectories)):
        dl = read_trajectory_and_extract_features(a2g, raw_traj_files[i])
        for do in dl:
            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(do, protocol=-1))
            txn.commit()

            idx += 1

    db.sync()
    db.close()
