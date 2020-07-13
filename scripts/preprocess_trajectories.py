"""
Reads a folder of Trajectory files and saves an LMDB with extracted features.
"""

import argparse
import os
import pickle
import random

import ase.io
import lmdb
import numpy as np
from ase.io.trajectory import Trajectory
from tqdm import tqdm

import torch
from ocpmodels.preprocessing import AtomsToGraphs


def read_trajectory_and_extract_features(a2g, traj_path):
    traj = ase.io.read(traj_path, ":")
    traj_len = len(traj)
    sample_count = int(0.1 * traj_len)
    sample_idx = random.sample(range(traj_len), sample_count)
    images = []
    for idx in sample_idx:
        uid = os.path.join(traj_path, f" {idx}\n")
        if uid not in sampled_unique_ids:
            sampled_unique_ids.append(uid)
            images.append(traj[idx])
    return a2g.convert_all(images, disable_tqdm=True)


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
        "--size",
        type=int,
        default=None,
        required=True,
        help="Size of dataset to construct",
    )
    parser.add_argument(
        "--filter",
        type=float,
        default=1.0e9,
        help="Max force to filter out, default: no filter",
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
    ids_log = open(os.path.join(args.out_path, "data_log.txt"), "w")
    sampled_unique_ids = []
    pbar = tqdm(total=args.size)

    idx = 0
    while idx <= args.size:
        try:
            # randomly select trajectory
            i = random.randrange(0, num_trajectories)
            # randomly sample subset of trajectory
            dl = read_trajectory_and_extract_features(a2g, raw_traj_files[i])
            for do in dl:
                # filter out images with excessivley large forces, if applicable
                if torch.max(torch.abs(do.force)).item() <= args.filter:
                    txn = db.begin(write=True)
                    txn.put(
                        f"{idx}".encode("ascii"), pickle.dumps(do, protocol=-1)
                    )
                    txn.commit()
                    idx += 1
                    pbar.update(1)
        except Exception:
            continue

    db.sync()
    db.close()
    pbar.close()
    ids_log.writelines(sampled_unique_ids)
