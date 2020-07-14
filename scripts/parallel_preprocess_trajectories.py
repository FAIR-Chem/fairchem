"""
Reads a txt file with paths to ASE trajectories and
creates LMDB files with extracted graph features.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import random

import ase.io
import lmdb
import torch
from tqdm import tqdm

from ocpmodels.preprocessing import AtomsToGraphs


def write_images_to_lmdb(mp_arg):
    a2g, db_path, traj_paths, sampled_unique_ids, idx = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    try:
        traj_idx = random.randrange(0, len(traj_paths))
        dl, sampled_unique_ids = read_trajectory_and_extract_features(
            a2g, traj_paths[traj_idx], sampled_unique_ids
        )
        for do in dl:
            # filter out images with excessively large forces, if applicable
            if (
                torch.max(torch.abs(do.force)).item()
                <= args.force_filter_threshold
            ):
                txn = db.begin(write=True)
                txn.put(
                    f"{idx}".encode("ascii"), pickle.dumps(do, protocol=-1)
                )
                txn.commit()
                idx += 1
    except Exception:
        pass

    db.sync()
    db.close()

    return sampled_unique_ids, idx


def read_trajectory_and_extract_features(a2g, traj_path, sampled_unique_ids):
    traj = ase.io.read(traj_path, ":")
    traj_len = len(traj)
    # 0.1 heuristic should work fine as long as args.size is greater than
    # 20 * num_trajectories (each trajectory has 200 images on average).
    sample_count = int(0.1 * traj_len)
    sample_idx = random.sample(range(traj_len), sample_count)
    images = []
    for idx in sample_idx:
        uid = os.path.join(traj_path, f" {idx}\n")
        if uid not in sampled_unique_ids:
            sampled_unique_ids.append(uid)
            images.append(traj[idx])
    return a2g.convert_all(images, disable_tqdm=True), sampled_unique_ids


def chunk_list(lst, num_splits):
    n = max(1, len(lst) // num_splits)
    return [lst[i : i + n] for i in range(0, len(lst), n)]


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
        "--num-workers",
        type=int,
        default=1,
        required=True,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        required=True,
        help="No. of images in the dataset",
    )
    parser.add_argument(
        "--force-filter-threshold",
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
        r_fixed=True,
    )

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%03d.lmdb" % i)
        for i in range(args.num_workers)
    ]

    # Chunk the trajectories into args.num_workers splits
    chunked_traj_files = chunk_list(raw_traj_files, args.num_workers)

    # Extract features
    sampled_unique_ids, idx = [[]] * args.num_workers, [0] * args.num_workers
    pbar = tqdm(total=args.size)
    pool = mp.Pool(args.num_workers)
    while sum(idx) <= args.size:
        mp_args = [
            (
                a2g,
                db_paths[i],
                chunked_traj_files[i],
                sampled_unique_ids[i],
                idx[i],
            )
            for i in range(args.num_workers)
        ]
        op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))
        sampled_unique_ids, idx = list(op[0]), list(op[1])
        pbar.update(sum(idx) - pbar.last_print_n)

    # Log sampled image, trajectory trace
    for i in range(args.num_workers):
        ids_log = open(
            os.path.join(args.out_path, "data_log.%03d.txt" % i), "w"
        )
        ids_log.writelines(sampled_unique_ids[i])

    pbar.close()
