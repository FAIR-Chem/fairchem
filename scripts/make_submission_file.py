"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import glob
import os

import numpy as np

SPLITS = {
    "OC20": ["id", "ood_ads", "ood_cat", "ood_both"],
    "OC22": ["id", "ood"],
}


def write_is2re_relaxations(args, dataset):
    import ase.io
    from tqdm import tqdm

    submission_file = {}

    if not args.hybrid:
        for split in SPLITS[dataset]:
            ids = []
            energies = []
            systems = glob.glob(os.path.join(vars(args)[split], "*.traj"))
            for system in tqdm(systems):
                sid, _ = os.path.splitext(os.path.basename(system))
                ids.append(str(sid))
                # Read the last frame in the ML trajectory. Modify "-1" if you wish to modify which frame to use.
                traj = ase.io.read(system, "199")
                energies.append(traj.get_potential_energy())

            submission_file[f"{split}_ids"] = np.array(ids)
            submission_file[f"{split}_energy"] = np.array(energies)

    else:
        for split in SPLITS[dataset]:
            preds = np.load(vars(args)[split])
            ids = []
            energies = []
            for sid, energy in zip(preds["ids"], preds["energy"]):
                sid = sid.split("_")[0]
                ids.append(sid)
                energies.append(energy)

            submission_file[f"{split}_ids"] = np.array(ids)
            submission_file[f"{split}_energy"] = np.array(energies)

    np.savez_compressed(args.out_path, **submission_file)


def write_predictions(args, dataset):
    if args.is2re_relaxations:
        write_is2re_relaxations(args, dataset=dataset)
    else:
        submission_file = {}

        for split in SPLITS[dataset]:
            res = np.load(vars(args)[split], allow_pickle=True)
            verify_dtype(res, dataset)
            contents = res.files
            for i in contents:
                key = "_".join([split, i])
                submission_file[key] = res[i]

        np.savez_compressed(args.out_path, **submission_file)


def verify_dtype(preds, dataset):
    if dataset == "OC22":
        if "energy" in preds:
            assert preds["energy"].dtype in [
                np.float32,
                np.float64,
            ], "Predictions written in the wrong precision. Ensure `total_energy` flag is True in the config."
        if "forces" in preds:
            assert preds["forces"].dtype in [
                np.float32,
                np.float64,
            ], "Predictions written in the wrong precision. Ensure `total_energy` flag is True in the config."


def main(args):
    if args.oc22:
        for split in SPLITS["OC22"]:
            assert vars(args).get(split), f"Missing {split} split for OC22"
        dataset = "OC22"
    else:
        for split in SPLITS["OC20"]:
            assert vars(args).get(split), f"Missing {split} split for OC20"
        dataset = "OC20"

    if not args.out_path.endswith(".npz"):
        args.out_path = args.out_path + ".npz"

    write_predictions(args, dataset=dataset)
    print(f"Results saved to {args.out_path} successfully.")


if __name__ == "__main__":
    """
    Create a submission file for evalAI. Ensure that for the task you are
    submitting for you have generated results files on each of the splits:
        OC20: id, ood_ads, ood_cat, ood_both
        OC22: id, ood

    Results file can be obtained as follows for the various tasks:

    S2EF: config["mode"] = "predict"
    IS2RE: config["mode"] = "predict"
    IS2RS: config["mode"] = "run-relaxations" and config["task"]["write_pos"] = True

    Use this script to join the results files (4 for OC20, 2 for OC22) in the format evalAI expects
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
    parser.add_argument(
        "--id", help="Path to ID results. Required for OC20 and OC22."
    )
    parser.add_argument(
        "--ood-ads", help="Path to OOD-Ads results. Required only for OC20."
    )
    parser.add_argument(
        "--ood-cat", help="Path to OOD-Cat results. Required only for OC20."
    )
    parser.add_argument(
        "--ood-both", help="Path to OOD-Both results. Required only for OC20."
    )
    parser.add_argument(
        "--ood", help="Path to OOD OC22 results. Required only for OC22."
    )
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
    parser.add_argument(
        "--oc22", action="store_true", help="Write OC22 prediction files."
    )

    args = parser.parse_args()
    main(args)
