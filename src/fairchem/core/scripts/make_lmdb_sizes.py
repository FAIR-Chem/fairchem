"""
This script provides the functionality to generate metadata.npz files necessary
for load_balancing the DataLoader.

"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import warnings

import numpy as np
from tqdm import tqdm

from fairchem.core.common.typing import assert_is_instance
from fairchem.core.datasets.lmdb_dataset import LmdbDataset


def get_data(index):
    data = dataset[index]
    natoms = data.natoms
    neighbors = None
    if hasattr(data, "edge_index") and data.edge_index is not None:
        neighbors = data.edge_index.shape[1]

    return index, natoms, neighbors


def make_lmdb_sizes(args) -> None:
    path = assert_is_instance(args.data_path, str)
    global dataset
    dataset = LmdbDataset({"src": path})
    if os.path.isdir(path):
        outpath = os.path.join(path, "metadata.npz")
    elif os.path.isfile(path):
        outpath = os.path.join(os.path.dirname(path), "metadata.npz")

    output_indices = range(len(dataset))

    pool = mp.Pool(assert_is_instance(args.num_workers, int))
    outputs = list(tqdm(pool.imap(get_data, output_indices), total=len(output_indices)))

    indices = []
    natoms = []
    neighbors = []
    for i in outputs:
        indices.append(i[0])
        natoms.append(i[1])
        neighbors.append(i[2])

    _sort = np.argsort(indices)
    sorted_natoms = np.array(natoms, dtype=np.int32)[_sort]
    if None in neighbors:
        warnings.warn(
            f"edge_index information not found, {outpath} only supports atom-wise load balancing."
        )
        np.savez(outpath, natoms=sorted_natoms)
    else:
        sorted_neighbors = np.array(neighbors, dtype=np.int32)[_sort]
        np.savez(outpath, natoms=sorted_natoms, neighbors=sorted_neighbors)


def get_lmdb_sizes_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Path to S2EF directory or IS2R* .lmdb file",
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Num of workers to parallelize across",
    )
    return parser


if __name__ == "__main__":
    parser = get_lmdb_sizes_parser()
    args: argparse.Namespace = parser.parse_args()
    make_lmdb_sizes(args)
