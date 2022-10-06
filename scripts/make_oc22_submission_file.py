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
        for idx, split in enumerate(["id", "ood"]):
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
        for idx, split in enumerate(["id", "ood"]):
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

    for idx, split in enumerate(["id", "ood"]):
        res = np.load(paths[idx], allow_pickle=True)
        assert np.issubdtype(
            res["energy"].dtype, np.float32
        ), "predictions written in the wrong precision, check total_energy flag is True in dataset config"
        assert np.issubdtype(
            res["forces"].dtype, np.float32
        ), "predictions written in the wrong precision, check total_energy flag is True in dataset config"
        contents = res.files
        for i in contents:
            key = "_".join([split, i])
            submission_file[key] = res[i]

    np.savez_compressed(filename, **submission_file)


def main(args):
    id_path = args.id
    ood_path = args.ood

    paths = [id_path, ood_path]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Path to ID results")
    parser.add_argument("--ood", help="Path to OOD results")
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
