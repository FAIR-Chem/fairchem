"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Reads a txt file with paths to ASE trajectories and creates LMDB files with
extracted graphs for predicting relaxed state energies from initial structures.
"""

import argparse
import multiprocessing as mp
import os
import pickle
import random

import lmdb
import torch
from ase import Atoms
from ase.io.trajectory import Trajectory
from tqdm import tqdm

from ocpmodels.preprocessing import AtomsToGraphs


def read_trajectory_and_extract_features(a2g, traj_path):
    traj = Trajectory(traj_path)
    images = [traj[0], traj[-1]]
    return a2g.convert_all(images, disable_tqdm=True)


if __name__ == "__main__":
    # Parse a few arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj-ids",
        default=None,
        required=True,
        help="Path to file containing system ids (one per line)",
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
        help="Size of the dataset",
    )
    parser.add_argument(
        "--inputid-to-adbulkid",
        type=str,
        default="/checkpoint/electrocatalysis/relaxations/mapping_old/pickled_mapping/mapping_inputid_to_input_bulk_adbulk_paths.pkl",
        help="Path to file containing input id to adbulk id mapping",
    )
    parser.add_argument(
        "--traj-paths",
        type=str,
        default="/checkpoint/electrocatalysis/relaxations/mapping_old/pickled_mapping/mapping_inputid_to_bulkid_bulkenergy_adbulkid_adbulkpath.pkl",
        help="Path to file containing adbulk id to path mapping",
    )
    parser.add_argument(
        "--adslab-ref",
        type=str,
        default="/checkpoint/electrocatalysis/relaxations/mapping/new_adslab_ref_energies_09_22_20.pkl",
        help="Path to reference energies, default: None",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="/checkpoint/electrocatalysis/relaxations/mapping_old/pickled_mapping/adslab_tags_full.pkl",
        help="Path to tags (0/1/2 for bulk / slab / adsorbates), default: None",
    )
    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Store LMDBs for test data, does not save targets",
    )

    args = parser.parse_args()

    # Read the txt file with trajectory paths.
    with open(args.traj_ids, "r") as f:
        raw_traj_ids = f.read().splitlines()

    with open(args.inputid_to_adbulkid, "rb") as f:
        inputid_to_adbulkid = pickle.load(f)

    raw_traj_files, errors1, errors2 = [], [], []
    with open(args.traj_paths, "rb") as f:
        d = pickle.load(f)
        for i in raw_traj_ids:
            try:
                adbulkid = inputid_to_adbulkid[i]["mapped_adbulk_id"]
            except Exception as e:
                print(str(e))
                errors1.append(i)
                continue

            try:
                raw_traj_files.append(d[adbulkid]["adbulk_path"])
            except Exception as e:
                print(str(e))
                errors2.append(i)
                continue

    num_trajectories = len(raw_traj_files)

    print(
        "### Found %d trajectories in %s" % (num_trajectories, args.traj_ids)
    )

    if len(errors1) > 0 or len(errors2) > 0:
        import pdb

        pdb.set_trace()

    # Read the adslab reference energies to subtract from potential energies.
    with open(args.adslab_ref, "rb") as f:
        adslab_ref = pickle.load(f)

    # Read tag information for each atom.
    with open(args.tags, "rb") as f:
        sysid_to_tags = pickle.load(f)

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=not args.test_data,
        r_forces=not args.test_data,
        r_distances=True,
        r_fixed=True,
    )

    # Create output directory if it doesn't exist.
    os.makedirs(os.path.join(args.out_path), exist_ok=True)

    # Initialize lmdb paths
    db_path = os.path.join(args.out_path, "data.lmdb")
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # Prune the trajectories list based on args.size
    pruned_traj_files = raw_traj_files[: args.size]

    # Extract features
    idx = 0
    removed = []
    for i, traj_path in tqdm(enumerate(pruned_traj_files)):
        try:
            dl = read_trajectory_and_extract_features(a2g, traj_path)
        except Exception as e:
            print(str(e), traj_path)
            continue

        randomid = os.path.split(traj_path)[1].split(".")[0]
        sid = int(randomid.split("random")[1])
        try:
            dl[0].sid = sid
            dl[0].tags = torch.LongTensor(sysid_to_tags[randomid])
            if not args.test_data:
                dl[0].y_init = dl[0].y - adslab_ref[randomid]
                dl[0].y_relaxed = dl[1].y - adslab_ref[randomid]
                dl[0].pos_relaxed = dl[1].pos
                del dl[0].y
        except Exception as e:
            import pdb

            pdb.set_trace()
            print(str(e), traj_path)
            continue

        if not args.test_data:
            if dl[0].y_relaxed > 10 or dl[0].y_relaxed < -10:
                print(traj_path, dl[0].y_relaxed)
                removed.append(traj_path + " %f\n" % dl[0].y_relaxed)
                continue

        # If no neighbors, skip.
        if dl[0].edge_index.shape[1] == 0:
            print("no neighbors", traj_path)
            continue

        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(dl[0], protocol=-1))
        txn.commit()
        idx += 1

    db.sync()
    db.close()

    # Log removed trajectories and their adsorption energies
    print("### Filtered out %d trajectories" % len(removed))
    removed_log = open(os.path.join(args.out_path, "removed.txt"), "w")
    removed_log.writelines(removed)
