"""
Creates LMDB files with extracted graph features from provided *.extxyz files
for the S2EF task.
"""

import argparse
import glob
import multiprocessing as mp
import os
import pickle

import ase.io
import lmdb
import numpy as np
import torch
from tqdm import tqdm

from ocpmodels.preprocessing import AtomsToGraphs


def write_images_to_lmdb(mp_arg) -> None:
    a2g, db_path, samples, pid = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=len(samples),
        position=pid,
        desc="Preprocessing data into LMDBs",
    )
    idx = 0
    for sample in samples:
        ml_relaxed = ase.io.read(sample, "-1")
        data_object = a2g.convert(ml_relaxed)

        sid, _ = os.path.splitext(os.path.basename(sample))
        fid = -1
        # add atom tags
        data_object.tags = torch.LongTensor(ml_relaxed.get_tags())
        data_object.sid = int(sid)
        data_object.fid = fid

        txn = db.begin(write=True)
        txn.put(
            f"{idx}".encode("ascii"),
            pickle.dumps(data_object, protocol=-1),
        )
        txn.commit()
        idx += 1
        pbar.update(1)

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()


def main(args, split) -> None:
    systems = glob.glob(f"{eval(f'args.{split}')}/*.traj")

    systems_chunked = np.array_split(systems, args.num_workers)

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_fixed=True,
        r_edges=True,
    )

    # Create output directory if it doesn't exist.
    out_path = f"{args.out_path}_{split}"
    os.makedirs(out_path, exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(out_path, "data.%04d.lmdb" % i)
        for i in range(args.num_workers)
    ]

    pool = mp.Pool(args.num_workers)
    mp_args = [
        (
            a2g,
            db_paths[i],
            systems_chunked[i],
            i,
        )
        for i in range(args.num_workers)
    ]
    list(pool.imap(write_images_to_lmdb, mp_args))
    pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        required=True,
        help="Path to ID trajectories",
    )
    parser.add_argument(
        "--ood-ads",
        required=True,
        help="Path to OOD-Ads trajectories",
    )
    parser.add_argument(
        "--ood-cat",
        required=True,
        help="Path to OOD-Cat trajectories",
    )
    parser.add_argument(
        "--ood-both",
        required=True,
        help="Path to OOD-Both trajectories",
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
        help="No. of feature-extracting processes.",
    )

    args: argparse.Namespace = parser.parse_args()

    for split in ["id", "ood_ads", "ood_cat", "ood_both"]:
        main(args, split)
