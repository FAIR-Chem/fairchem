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
import numpy as np
import torch
from tqdm import tqdm

from ocpmodels.preprocessing import AtomsToGraphs


def write_images_to_lmdb(mp_arg):
    a2g, db_path, randomids, sampled_unique_ids, idx = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for randomid in randomids:
        try:
            (
                dl,
                process_samples,
                idx_log,
            ) = read_trajectory_and_extract_features(
                a2g, randomid, sampled_unique_ids
            )
            for i, do in enumerate(dl):
                # filter out images with excessively large forces, if applicable
                if torch.max(torch.abs(do.force)).item() <= args.filter:
                    # add atom tags
                    if args.tags:
                        do.tags = torch.LongTensor(tags_map[randomid])
                    # subtract off reference energy
                    if args.ref_energy:
                        do.y -= adslab_ref[randomid]
                    txn = db.begin(write=True)
                    txn.put(
                        f"{idx}".encode("ascii"), pickle.dumps(do, protocol=-1)
                    )
                    txn.commit()
                    sampled_unique_ids.append(process_samples[i])
                    idx += 1
        except Exception:
            pass

    db.sync()
    db.close()

    return sampled_unique_ids, idx


def read_trajectory_and_extract_features(a2g, randomid, sampled_unique_ids):
    adbulk_id = map_to_adbulk[randomid]["mapped_adbulk_id"]
    traj_path = adbulk_to_trajpath[adbulk_id]["adbulk_path"]
    traj = ase.io.read(traj_path, ":")
    traj_len = len(traj)
    sample_idx = range(traj_len)
    process_samples = []
    images = []
    idx_log = []
    for idx in sample_idx:
        uid = "{},{},{}\n".format(traj_path, idx, traj_len)
        if uid not in sampled_unique_ids:
            process_samples.append(uid)
            images.append(traj[idx])
            idx_log.append((idx, traj_len))
    return (
        a2g.convert_all(images, disable_tqdm=True),
        process_samples,
        idx_log,
    )


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
        "--filter",
        type=float,
        default=1.0e9,
        help="Max force to filter out, default: no filter",
    )
    parser.add_argument(
        "--ref-energy", action="store_true", help="Subtract reference energies"
    )
    parser.add_argument("--tags", action="store_true", help="Add atom tags")
    parser.add_argument(
        "--chunk", type=int, help="Chunk to of inputs to preprocess"
    )

    args = parser.parse_args()

    # Read the txt file with trajectory paths.
    with open(os.path.join(args.data_path), "r") as f:
        input_ids = f.read().splitlines()
    num_trajectories = len(input_ids)

    with open(
        os.path.join(
            "/checkpoint/electrocatalysis/relaxations/mapping/pickled_mapping/mapping_inputid_to_input_bulk_adbulk_paths.pkl"
        ),
        "rb",
    ) as k:
        map_to_adbulk = pickle.load(k)

    with open(
        os.path.join(
            "/checkpoint/electrocatalysis/relaxations/mapping/pickled_mapping/mapping_inputid_to_bulkid_bulkenergy_adbulkid_adbulkpath.pkl"
        ),
        "rb",
    ) as l:
        adbulk_to_trajpath = pickle.load(l)

    if args.ref_energy:
        ref_path = "/checkpoint/electrocatalysis/relaxations/mapping/pickled_mapping/adslab_ref_energies_full.pkl"
        with open(os.path.join(ref_path), "rb") as g:
            adslab_ref = pickle.load(g)

    if args.tags:
        tag_path = "/checkpoint/electrocatalysis/relaxations/mapping/pickled_mapping/adslab_tags_full.pkl"
        with open(os.path.join(tag_path), "rb") as h:
            tags_map = pickle.load(h)

    print(
        "### Found %d trajectories in %s" % (num_trajectories, args.data_path)
    )

    # set random seed
    random.seed(348952)

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

    slurm_chunks = np.array_split(input_ids, 20)
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
    sampled_unique_ids, idx = [[]] * args.num_workers, [0] * args.num_workers
    pool = mp.Pool(args.num_workers)
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

    # Log sampled image, trajectory trace
    for j, i in enumerate(range(lmdb_start_idx, lmdb_end_idx)):
        ids_log = open(
            os.path.join(args.out_path, "data_log.%04d.txt" % i), "w"
        )
        ids_log.writelines(sampled_unique_ids[j])
