"""
Creates LMDB files with extracted graph features from provided *.extxyz files
for the S2EF task.
"""

import argparse
import glob
import multiprocessing as mp
import os
import pickle
import random
import sys

import ase.io
import lmdb
import numpy as np
import torch
from tqdm import tqdm

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
        traj_logs = open(sample, "r").read().splitlines()
        xyz_idx = os.path.splitext(os.path.basename(sample))[0]
        traj_path = os.path.join(args.data_path, f"{xyz_idx}.extxyz")
        traj_frames = ase.io.read(traj_path, ":")

        for i, frame in enumerate(traj_frames):
            frame_log = traj_logs[i].split(",")
            data_object = a2g.convert(frame)
            # add atom tags
            data_object.tags = torch.LongTensor(frame.get_tags())
            # subtract off reference energy
            if args.ref_energy:
                ref_energy = float(frame_log[2])
                data_object.y -= ref_energy

            txn = db.begin(write=True)
            txn.put(
                f"{idx}".encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
            txn.commit()
            idx += 1
            sampled_ids.append(",".join(frame_log[:2]) + "\n")

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return sampled_ids, idx


def chunk_list(lst, num_splits):
    n = max(1, len(lst) // num_splits)
    return [lst[i : i + n] for i in range(0, len(lst), n)]


if __name__ == "__main__":
    # Parse a few arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to dir containing *.extxyz and *.txt files",
    )
    parser.add_argument(
        "--out-path",
        required=True,
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB, ~10x storage requirement. Default: compute edge indices on-the-fly.",
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
    args = parser.parse_args()

    xyz_logs = glob.glob(os.path.join(args.data_path, "*.txt"))

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_fixed=True,
        r_edges=args.get_edges,
    )

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(args.out_path, "data.%04d.lmdb" % i)
        for i in range(args.num_workers)
    ]

    # Chunk the trajectories into args.num_workers splits
    chunked_txt_files = chunk_list(xyz_logs, args.num_workers)

    # Extract features
    sampled_ids, idx = [[]] * args.num_workers, [0] * args.num_workers
    pool = mp.Pool(args.num_workers)
    mp_args = [
        (
            a2g,
            db_paths[i],
            chunked_txt_files[i],
            sampled_ids[i],
            idx[i],
        )
        for i in range(args.num_workers)
    ]
    op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))
    sampled_ids, idx = list(op[0]), list(op[1])

    # Log sampled image, trajectory trace
    for j, i in enumerate(range(args.num_workers)):
        ids_log = open(
            os.path.join(args.out_path, "data_log.%04d.txt" % i), "w"
        )
        ids_log.writelines(sampled_ids[j])
