"""
Reads a txt file with paths to ASE trajectories and
creates LMDB files with extracted graph features.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import random

import numpy as np
from tqdm import tqdm

import ase.io
import lmdb
import torch
from ocpmodels.preprocessing import AtomsToGraphs


def write_images_to_lmdb(mp_arg):
    a2g, db_path, samples, sampled_ids, idx = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for sample in samples:
        try:
            splits = sample.split(",")
            traj_path = splits[0]
            traj_idx = int(splits[1])
            randomid = splits[-1]

            do = read_trajectory_and_extract_features(a2g, traj_path, traj_idx)
            # add atom tags
            if args.tags:
                do.tags = torch.LongTensor(tags_map[randomid])
            # subtract off reference energy
            if args.ref_energy:
                do.y -= adslab_ref[randomid]
            if do.edge_index.shape[1] == 0:
                print("no neighbors", traj_path)
                continue
            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(do, protocol=-1))
            txn.commit()
            idx += 1
            sampled_ids.append(",".join(splits[:3]) + "\n")
        except Exception:
            pass

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put(f"length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return sampled_ids, idx


def read_trajectory_and_extract_features(a2g, traj_path, traj_idx):
    image = ase.io.trajectory.Trajectory(traj_path)[traj_idx]
    return a2g.convert(image)


def chunk_list(lst, num_splits):
    n = max(1, len(lst) // num_splits)
    return [lst[i : i + n] for i in range(0, len(lst), n)]


if __name__ == "__main__":
    # Parse a few arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to txt file containing input ids (one per line)",
    )
    parser.add_argument(
        "--out-path",
        required=True,
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--ref-energy", action="store_true", help="Subtract reference energies"
    )
    parser.add_argument("--tags", action="store_true", help="Add atom tags")
    parser.add_argument(
        "--chunk", type=int, help="Chunk to of inputs to preprocess"
    )
    parser.add_argument(
        "--size", default=-1, type=int, help="Size of dataset to process"
    )

    args = parser.parse_args()

    # Read the txt file with trajectory paths.
    with open(os.path.join(args.data_path), "r") as f:
        if args.size != -1:
            input_ids = f.read().splitlines()[: args.size]
        else:
            input_ids = f.read().splitlines()
    num_trajectories = len(input_ids)

    if args.ref_energy:
        ref_path = "/checkpoint/electrocatalysis/relaxations/mapping/new_adslab_ref_energies_09_22_20.pkl"
        with open(os.path.join(ref_path), "rb") as g:
            adslab_ref = pickle.load(g)

    if args.tags:
        tag_path = "/checkpoint/electrocatalysis/relaxations/mapping_old/pickled_mapping/adslab_tags_full.pkl"
        with open(os.path.join(tag_path), "rb") as h:
            tags_map = pickle.load(h)

    print(
        "### Found %d trajectories in %s" % (num_trajectories, args.data_path)
    )

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_fixed=True,
    )

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Main chunks to be split across 40 nodes
    slurm_chunks = np.array_split(input_ids, 40)
    lmdb_start_idx = args.chunk * args.num_workers
    lmdb_end_idx = lmdb_start_idx + args.num_workers

    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i)
        for i in range(lmdb_start_idx, lmdb_end_idx)
    ]

    chunk_to_process = slurm_chunks[args.chunk]

    # Chunk the trajectories into args.num_workers splits
    chunked_traj_files = chunk_list(chunk_to_process, args.num_workers)

    # Extract features
    sampled_ids, idx = [[]] * args.num_workers, [0] * args.num_workers
    pool = mp.Pool(args.num_workers)
    mp_args = [
        (a2g, db_paths[i], chunked_traj_files[i], sampled_ids[i], idx[i],)
        for i in range(args.num_workers)
    ]
    op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))
    sampled_ids, idx = list(op[0]), list(op[1])

    # Log sampled image, trajectory trace
    for j, i in enumerate(range(lmdb_start_idx, lmdb_end_idx)):
        ids_log = open(
            os.path.join(args.out_path, "data_log.%04d.txt" % i), "w"
        )
        ids_log.writelines(sampled_ids[j])
