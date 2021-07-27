"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import glob
import os

import numpy as np


def write_is2re_relaxations(paths, filename, hybrid):
    import ase.io
    from tqdm import tqdm

    submission_file = {}

    if not hybrid:
        for idx, split in enumerate(["id", "ood_ads", "ood_cat", "ood_both"]):
            ids = []
            energies = []
            systems = glob.glob(os.path.join(paths[idx], "*.traj"))
            for system in tqdm(systems):
                sid, _ = os.path.splitext(os.path.basename(system))
                ids.append(str(sid))
                traj = ase.io.read(system, "-1")
                energies.append(traj.get_potential_energy())

            submission_file[f"{split}_ids"] = np.array(ids)
            submission_file[f"{split}_energy"] = np.array(energies)

    else:
        for idx, split in enumerate(["id", "ood_ads", "ood_cat", "ood_both"]):
            preds = np.load(paths[idx])
            ids = []
            energies = []
            for sid, energy in zip(preds["ids"], preds["energy"]):
                sid = sid.split("_")[0]
                ids.append(sid)
                energies.append(energy)

            submission_file[f"{split}_ids"] = np.array(ids)
            submission_file[f"{split}_energy"] = np.array(energies)

    np.savez_compressed(filename, **submission_file)


def write_predictions(paths, filename):
    submission_file = {}

    for idx, split in enumerate(["id", "ood_ads", "ood_cat", "ood_both"]):
        res = np.load(paths[idx], allow_pickle=True)
        contents = res.files
        for i in contents:
            key = "_".join([split, i])
            submission_file[key] = res[i]

    np.savez_compressed(filename, **submission_file)


def main(args):
    id_path = args.id
    ood_ads_path = args.ood_ads
    ood_cat_path = args.ood_cat
    ood_both_path = args.ood_both

    paths = [id_path, ood_ads_path, ood_cat_path, ood_both_path]
    if not args.out_path.endswith(".npz"):
        args.out_path = args.out_path + ".npz"

    if not args.is2re_relaxations:
        write_predictions(paths, filename=args.out_path)
    else:
        write_is2re_relaxations(
            paths, filename=args.out_path, hybrid=args.hybrid
        )
    print(f"Results saved to {args.out_path} successfully.")


if __name__ == "__main__":
    """
    Create a submission file for evalAI. Ensure that for the task you are
    submitting for you have generated results files on each of the 4 splits -
    id, ood_ads, ood_cat, ood_both.

    Results file can be obtained as follows for the various tasks:

    S2EF: config["mode"] = "predict"
    IS2RE: config["mode"] = "predict"
    IS2RS: config["mode"] = "run-relaxations" and config["task"]["write_pos"] = True

    Use this script to join the 4 results files in the format evalAI expects
    submissions.

    If writing IS2RE predictions from relaxations, paths must be directories
    containg trajectory files. Additionally, --is2re-relaxations must be
    provided as a command line argument.

    If writing IS2RE predictions from hybrid relaxations (force only model +
    energy only model), paths must be the .npz S2EF prediction files.
    Additionally, --is2re-relaxations and --hybrid must be provided as a
    command line argument.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Path to ID results")
    parser.add_argument("--ood-ads", help="Path to OOD-Ads results")
    parser.add_argument("--ood-cat", help="Path to OOD-Cat results")
    parser.add_argument("--ood-both", help="Path to OOD-Both results")
    parser.add_argument("--out-path", help="Path to write predictions to.")
    parser.add_argument(
        "--is2re-relaxations",
        action="store_true",
        help="Write IS2RE results from trajectories. Paths specified correspond to directories containing .traj files.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Write IS2RE results from S2EF prediction files. Paths specified correspond to S2EF NPZ files.",
    )

    args = parser.parse_args()
    main(args)
