from __future__ import annotations

import argparse
import errno
import os

import ase
import ase.io
from ase.io.trajectory import TrajectoryReader


def compare_runs(path1, path2, reference_type, tol):
    if not os.path.exists(path1):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path1)
    if not os.path.exists(path2):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path2)

    if reference_type == "xml":
        atoms1 = ase.io.read(path1)
    elif reference_type == "traj":
        atoms1 = TrajectoryReader(path1)[-1]
    else:
        raise ValueError("Incorrect specification of type argument")

    atoms2 = ase.io.read(path2)

    pe1 = atoms1.get_potential_energy()
    pe2 = atoms2.get_potential_energy()

    if abs(pe1 - pe2) < tol:
        return True
    return False


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, help="Path of reference traj/xml file")
    parser.add_argument("--path2", type=str, help="Path of current vasprun.xml file")
    parser.add_argument(
        "--type",
        type=str,
        default="xml",
        help="Compare current vasprun.xml file with `xml` or `traj`",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance to compare potential energies",
    )
    return parser


def main(args):
    ans = compare_runs(args.path1, args.path2, args.type, args.tolerance)
    if ans:
        print("Passed: Converged to same structures")
    else:
        print("Failed: Converged to different relaxations")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
