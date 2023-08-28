import os
from logging import getLogger
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
from torch_geometric.data import Data
from typing_extensions import TypeVar

from .base import LmdbDataset

log = getLogger(__name__)

T = TypeVar("T", infer_variance=True)


class QM9Item(TypedDict):
    idx: torch.Tensor  # (1,)
    name: str
    z: torch.Tensor  # (n_atoms,)
    x: torch.Tensor  # (n_atoms, n_features)
    pos: torch.Tensor  # (n_atoms, 3)
    edge_index: torch.Tensor  # (2, n_edges)
    edge_attr: torch.Tensor  # (n_edges, n_edge_features)
    y: torch.Tensor  # (1, n_targets)


DOWNLOAD_URL = "https://data.pyg.org/datasets/qm9_v3.zip"
DOWNLOAD_FILENAME = "qm9_v3.zip"
NUM_VAL = 10_000
NUM_TEST = 10_000


class QM9(LmdbDataset[T]):
    targets = [
        "mu",  # dipole_moment
        "alpha",  # isotropic_polarizability
        "eps_HOMO",  # homo
        "eps_LUMO",  # lumo
        "delta_eps",  # homo_lumo_gap
        "R_2_Abs",  # electronic_spatial_extent
        "ZPVE",  # zpve
        "U_0",  # energy_U0
        "U",  # energy_U
        "H",  # enthalpy_H
        "G",  # free_energy
        "c_v",  # heat_capacity
        "U_0_ATOM",  # atomization_energy_U0
        "U_ATOM",  # atomization_energy_U
        "H_ATOM",  # atomization_enthalpy_H
        "G_ATOM",  # atomization_free_energy
        "A",  # rotational_constant_A
        "B",  # rotational_constant_B
        "C",  # rotational_constant_C
    ]

    @classmethod
    def download(
        cls,
        destination: str | Path,
        random_seed: int = 42,
    ):
        global DOWNLOAD_URL, DOWNLOAD_FILENAME

        root_path = Path(destination)
        root_path.mkdir(parents=True, exist_ok=True)

        # Make the raw data directory
        raw_dir = root_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Download the raw data
        raw_file = raw_dir / DOWNLOAD_FILENAME
        if not raw_file.exists():
            log.info("Downloading raw data")
            _ = os.system(f"wget -q -O {raw_file} {DOWNLOAD_URL}")

            # Unzip the raw data
            log.info("Unzipping raw data")
            _ = os.system(f"unzip {raw_file} -d {raw_dir}")
        else:
            log.info("Raw data already downloaded")

        # Load the raw data
        data = torch.load(raw_dir / "qm9_v3.pt")
        assert isinstance(data, list), f"{type(data)=} is not list"
        data = cast(list[QM9Item], data)

        # Create the splits
        num_indices = len(data)
        num_val = NUM_VAL
        num_test = NUM_TEST
        num_train = num_indices - num_val - num_test

        # Get the indices for each split (80/10/10 train/val/test)
        all_indices = np.arange(num_indices)
        np.random.RandomState(random_seed).shuffle(all_indices)

        split_indices = {
            "train": all_indices[:num_train],
            "val": all_indices[num_train : num_train + num_val],
            "test": all_indices[num_train + num_val :],
        }

        # Make sure the splits add up
        assert (
            len(split_indices["train"])
            + len(split_indices["val"])
            + len(split_indices["test"])
            == num_indices
        ), "Indices don't add up"
        split_sizes = {
            name: indices.shape[0] for name, indices in split_indices.items()
        }
        log.critical(f"Created the following splits: {split_sizes}")

        # Dump the indices to root/metadata/split_indices.npz
        cls.save_indices(split_indices, root_path)

        def _dump_idx(indices: np.ndarray):
            nonlocal data

            # Dump the systems
            for idx in indices:
                idx = int(idx)
                # Get the system and structure data
                system = data[idx]

                atomic_numbers = system["z"].clone().long()  # natoms
                pos = system["pos"].clone().float()  # natoms 3
                y = {
                    target: system["y"][0][i]
                    .clone()
                    .float()  # "y" is a tensor of shape (1, num_targets)
                    for i, target in enumerate(cls.targets)
                }

                data_object = Data(atomic_numbers=atomic_numbers, pos=pos, **y)
                yield data_object

        # Convert the raw data to LMDB
        log.info("Converting raw data to LMDB")

        # Make the processed data directory
        lmdb_path = root_path / "lmdb"
        lmdb_path.mkdir(parents=True, exist_ok=True)

        # Dump the frames
        for split, indices in split_indices.items():
            path = lmdb_path / split
            path.mkdir(parents=True, exist_ok=True)

            cls.dump_data(
                _dump_idx(indices),
                count=indices.shape[0],
                path=path,
                natoms_metadata_additional_path=root_path
                / "metadata"
                / split
                / "metadata.npz",
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    QM9.download(args.destination)
