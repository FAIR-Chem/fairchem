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


def gratoms_to_atoms(gratoms):
    atomsobj = Atoms(
        gratoms.symbols,
        gratoms.positions,
        None,
        gratoms.get_tags(),
        None,
        None,
        None,
        None,
        None,
        gratoms.cell,
        gratoms.pbc,
        None,
        gratoms.constraints,
        gratoms.info,
    )
    return atomsobj


def get_tags(data, randomid):
    mdata_path = os.path.join(sysid_mappings[randomid], "metadata.pkl")
    sort_path = os.path.join(sysid_mappings[randomid], "ase-sort.dat")
    k = open(mdata_path, "rb")
    metadata = pickle.load(k)
    k.close()
    sorts = []
    with open(sort_path, "r") as f:
        for line in f:
            sort, resort = line.split()
            sorts.append(int(sort))
    # pre-vasp sorted structure
    input_atoms = gratoms_to_atoms(metadata["adsorbed_bulk_atomsobject"])
    # sort to post-vasp structure
    input_atoms = input_atoms[sorts]
    # sanity check indices match up
    assert (
        input_atoms.get_atomic_numbers().all()
        == data.atomic_numbers.numpy().all()
    )
    tags = torch.tensor(input_atoms.get_tags())
    return tags


def read_trajectory_and_extract_features(a2g, traj_path):
    traj = Trajectory(traj_path)
    images = [traj[0], traj[-1]]
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
        help="Size of the dataset",
    )
    parser.add_argument(
        "--adslab-ref",
        type=str,
        default="/checkpoint/mshuaibi/mappings/adslab_ref_energies_ocp728k.pkl",
        help="Path to reference energies, default: None",
    )

    args = parser.parse_args()

    # Read the txt file with trajectory paths.
    with open(os.path.join(args.traj_paths_txt), "r") as f:
        raw_traj_files = f.read().splitlines()
    num_trajectories = len(raw_traj_files)

    with open(os.path.join(args.adslab_ref), "rb") as g:
        adslab_ref = pickle.load(g)

    with open(
        "/checkpoint/mshuaibi/mappings/sysid_to_bulkads_dir.pkl", "rb"
    ) as h:
        sysid_mappings = pickle.load(h)

    print(
        "### Found %d trajectories in %s"
        % (num_trajectories, args.traj_paths_txt)
    )

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=12,
        radius=6,
        r_energy=True,
        r_forces=True,
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
    for i, traj_path in tqdm(enumerate(pruned_traj_files)):
        try:
            dl = read_trajectory_and_extract_features(a2g, traj_path)
        except Exception as e:
            print(str(e), traj_path)
            continue

        randomid = os.path.split(traj_path)[1].split(".")[0]
        try:
            dl[0].tags = get_tags(dl[0], randomid)
            dl[0].y_init = dl[0].y - adslab_ref[randomid]
            dl[0].y_relaxed = dl[1].y - adslab_ref[randomid]
            dl[0].pos_relaxed = dl[1].pos
            del dl[0].y
        except Exception as e:
            print(str(e), traj_path)
            continue

        if dl[0].y_relaxed > 50 or dl[0].y_relaxed < -50:
            continue

        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(dl[0], protocol=-1))
        txn.commit()
        idx += 1

    db.sync()
    db.close()
