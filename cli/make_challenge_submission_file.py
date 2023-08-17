"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


ONLY for use in the NeurIPS 2021 Open Catalyst Challenge. For all other submissions
please use make_submission_file.py.
"""

import argparse
import glob
import os

import numpy as np


def write_is2re_relaxations(path: str, filename: str, hybrid) -> None:
    import ase.io
    from tqdm import tqdm

    submission_file = {}

    if not hybrid:
        ids = []
        energies = []
        systems = glob.glob(os.path.join(path, "*.traj"))
        for system in tqdm(systems):
            sid, _ = os.path.splitext(os.path.basename(system))
            ids.append(str(sid))
            traj = ase.io.read(system, "-1")
            energies.append(traj.get_potential_energy())

        submission_file["challenge_ids"] = np.array(ids)
        submission_file["challenge_energy"] = np.array(energies)

    else:
        preds = np.load(path)
        ids = []
        energies = []
        for sid, energy in zip(preds["ids"], preds["energy"]):
            sid = sid.split("_")[0]
            ids.append(sid)
            energies.append(energy)

        submission_file["challenge_ids"] = np.array(ids)
        submission_file["challenge_energy"] = np.array(energies)

    np.savez_compressed(filename, **submission_file)


def write_predictions(path: str, filename: str) -> None:
    submission_file = {}

    res = np.load(path, allow_pickle=True)
    contents = res.files
    for i in contents:
        key = "_".join(["challenge", i])
        submission_file[key] = res[i]

    np.savez_compressed(filename, **submission_file)


def main(args: argparse.Namespace) -> None:
    path = args.path

    if not args.out_path.endswith(".npz"):
        args.out_path = args.out_path + ".npz"

    if not args.is2re_relaxations:
        write_predictions(path, filename=args.out_path)
    else:
        write_is2re_relaxations(
            path, filename=args.out_path, hybrid=args.hybrid
        )
    print(f"Results saved to {args.out_path} successfully.")


if __name__ == "__main__":
    """
    Create a submission file for the NeurIPS 2021 Open Catalyst Challenge.

    Results file can be obtained as follows for the various tasks:

    S2EF: config["mode"] = "predict"
    IS2RE: config["mode"] = "predict"
    IS2RS: config["mode"] = "run-relaxations" and config["task"]["write_pos"] = True

    Use this script to write your results files in the format evalAI expects
    submissions.

    If writing IS2RE predictions from relaxations, the path specified must be a
    directory containg trajectory (.traj) files. Additionally, --is2re-relaxations must be
    provided as a command line argument.

    If writing IS2RE predictions from hybrid relaxations (force only model +
    energy only model), paths must be the .npz S2EF prediction files.
    Additionally, --is2re-relaxations and --hybrid must be provided as a
    command line argument.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to results")
    parser.add_argument("--out-path", help="Path to write predictions to.")
    parser.add_argument(
        "--is2re-relaxations",
        action="store_true",
        help="Write IS2RE results from trajectories. Path specified must be a directory containing .traj files.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Write IS2RE results from S2EF prediction files. Path specified must be a S2EF NPZ file.",
    )

    args: argparse.Namespace = parser.parse_args()
    main(args)
