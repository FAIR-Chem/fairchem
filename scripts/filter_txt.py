"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import multiprocessing as mp
import os
import pickle
import random
import sys

import ase.io
import numpy as np
from tqdm import tqdm


def filter_files(input_id, tol=50):
    adsorbate_id = map_to_adbulk[input_id]["mapped_adbulk_id"]
    path = adbulk_to_path[adsorbate_id]["adbulk_path"]
    images = ase.io.trajectory.Trajectory(path)
    traj_len = len(images)
    to_write = []
    to_delete = []
    for i, image in enumerate(images):
        max_force = np.max(np.abs(image.get_forces(apply_constraint=False)))
        name = f"{path},{i},{traj_len},{input_id}"
        if max_force < tol:
            to_write.append(name)
        else:
            to_delete.append(name)
    return to_write, to_delete


if __name__ == "__main__":
    data_idx = int(sys.argv[1])
    datasets = [
        "train",
        "val_is",
        "val_oos_ads",
        "val_oos_bulk",
        "val_oos_ads_bulk",
        "test_is",
        "test_oos_ads",
        "test_oos_bulk",
        "test_oos_ads_bulk",
    ]
    data = datasets[data_idx]

    k = open(
        f"/checkpoint/electrocatalysis/relaxations/mapping/final_splits_with_adbulk_ids/{data}.txt",
        "r",
    )
    systems = k.read().splitlines()
    j = open(
        "/checkpoint/electrocatalysis/relaxations/mapping/pickled_mapping/adslab_ref_energies_full.pkl",
        "rb",
    )
    ref = pickle.load(j)
    lp = open(
        "/checkpoint/electrocatalysis/relaxations/mapping/pickled_mapping/mapping_inputid_to_input_bulk_adbulk_paths.pkl",
        "rb",
    )
    map_to_adbulk = pickle.load(lp)
    m = open(
        "/checkpoint/electrocatalysis/relaxations/mapping/pickled_mapping/mapping_inputid_to_bulkid_bulkenergy_adbulkid_adbulkpath.pkl",
        "rb",
    )
    adbulk_to_path = pickle.load(m)

    pool = mp.Pool(60)
    outputs = list(tqdm(pool.imap(filter_files, systems), total=len(systems)))

    to_write = []
    to_delete = []
    for output in outputs:
        to_write += output[0]
        to_delete += output[1]

    # shuffle data
    random.seed(8472589)
    random.shuffle(to_write)

    q = open(f"{data}.txt", "w")
    p = open(f"deleted/{data}_del.txt", "w")

    for i in to_write:
        q.write(i + "\n")
    for j in to_delete:
        p.write(j + "\n")
