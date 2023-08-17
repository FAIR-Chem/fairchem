"""
Creates LMDB files with extracted graph features from provided *.extxyz files
for the S2EF task.
"""

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


def write_images_to_lmdb(mp_arg):
    (
        a2g,
        db_path,
        samples,
        sampled_ids,
        idx,
        pid,
        data_path,
        ref_energy,
        test_data,
    ) = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=5000 * len(samples),
        position=pid,
        desc="Preprocessing data into LMDBs",
    )
    for sample in samples:
        traj_logs = open(sample, "r").read().splitlines()
        xyz_idx = os.path.splitext(os.path.basename(sample))[0]
        traj_path = os.path.join(data_path, f"{xyz_idx}.extxyz")
        traj_frames = ase.io.read(traj_path, ":")

        for i, frame in enumerate(traj_frames):
            frame_log = traj_logs[i].split(",")
            sid = int(frame_log[0].split("random")[1])
            fid = int(frame_log[1].split("frame")[1])
            data_object = a2g.convert(frame)
            # add atom tags
            data_object.tags = torch.LongTensor(frame.get_tags())
            data_object.sid = sid
            data_object.fid = fid
            # subtract off reference energy
            if ref_energy and not test_data:
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
            pbar.update(1)

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return sampled_ids, idx


def main(
    data_path: str,
    out_path: str,
    num_workers: int,
    get_edges: bool,
    ref_energy: bool,
    test_data: bool,
) -> None:
    xyz_logs = glob.glob(os.path.join(data_path, "*.txt"))
    if not xyz_logs:
        raise RuntimeError("No *.txt files found. Did you uncompress?")
    if num_workers > len(xyz_logs):
        num_workers = len(xyz_logs)

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=not test_data,
        r_forces=not test_data,
        r_fixed=True,
        r_distances=False,
        r_edges=get_edges,
    )

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(out_path), exist_ok=True)

    # Initialize lmdb paths
    db_paths = [
        os.path.join(out_path, "data.%04d.lmdb" % i)
        for i in range(num_workers)
    ]

    # Chunk the trajectories into num_workers splits
    chunked_txt_files = np.array_split(xyz_logs, num_workers)

    # Extract features
    sampled_ids, idx = [[]] * num_workers, [0] * num_workers

    pool = mp.Pool(num_workers)
    mp_args = [
        (
            a2g,
            db_paths[i],
            chunked_txt_files[i],
            sampled_ids[i],
            idx[i],
            i,
            data_path,
            ref_energy,
            test_data,
        )
        for i in range(num_workers)
    ]
    op = list(zip(*pool.imap(write_images_to_lmdb, mp_args)))
    sampled_ids, idx = list(op[0]), list(op[1])

    # Log sampled image, trajectory trace
    for j, i in enumerate(range(num_workers)):
        ids_log = open(os.path.join(out_path, "data_log.%04d.txt" % i), "w")
        ids_log.writelines(sampled_ids[j])
