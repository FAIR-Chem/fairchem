import os
from logging import getLogger
from pathlib import Path
from typing import Any, TypedDict

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from typing_extensions import TypeVar, override

from ...modules.transforms.units import update_units_transform
from .base import LmdbDataset

log = getLogger(__name__)

T = TypeVar("T", infer_variance=True)

TRAIN_SIZE = 0.8
TEST_SIZE = 0.1


def _assert_dataset(dataset: Any) -> h5py.Dataset:
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{dataset=} is not a dataset")

    return dataset


class DatasetInfo(TypedDict):
    hdf5: str


class SPICE(LmdbDataset[T]):
    datasets: dict[str, DatasetInfo] = {
        "dipeptides": {"hdf5": "https://fm-datasets.s3.amazonaws.com/dipeptides.h5"},
        "solvated_amino_acids": {
            "hdf5": "https://fm-datasets.s3.amazonaws.com/solvated_amino_acids.h5"
        },
    }

    @classmethod
    def download(
        cls,
        destination: str | Path,
        dataset: str,
        random_seed: int = 42,
    ):
        global TRAIN_SIZE, TEST_SIZE

        if (info := cls.datasets.get(dataset)) is None:
            raise ValueError(f"{dataset=} is not a valid SPICE dataset name.")

        # Create root directory
        root_path = Path(destination)
        root_path.mkdir(parents=True, exist_ok=True)

        # Download dataset
        dl_path = root_path / "raw/"
        dl_path.mkdir(parents=True, exist_ok=True)

        # Download HDF5 file
        h5_path = dl_path / info["hdf5"].split("/")[-1]
        if not h5_path.exists():
            log.info(f"Downloading {info['hdf5']} to {h5_path}")
            _ = os.system(f"wget -q {info['hdf5']} -O {h5_path}")
        else:
            log.info(f"Found {h5_path}")

        # Load the h5 file
        with h5py.File(h5_path, "r") as f:
            # First, flatten the dataset to get a list of all conformers
            all_conformers: list[tuple[str, int]] = []
            for mol_idx, mol in f.items():
                if not isinstance(mol, h5py.Group):
                    continue
                conformations = mol["conformations"]
                if not isinstance(conformations, h5py.Dataset):
                    continue
                for conf_idx in range(conformations.shape[0]):
                    all_conformers.append((mol_idx, conf_idx))

            # Get the indices for each split (80/10/10 train/val/test)
            all_indices = np.arange(len(all_conformers))
            np.random.RandomState(random_seed).shuffle(all_indices)
            num_indices = len(all_indices)
            num_train = int(num_indices * TRAIN_SIZE)
            num_test = int(num_indices * TEST_SIZE)
            num_val = num_indices - num_train - num_test

            split_indices = {
                "train": all_indices[:num_train],
                "val": all_indices[num_train : num_train + num_val],
                "test": all_indices[num_train + num_val :],
            }

            # Make sure the splits add up
            assert (
                num_train + num_val + num_test == num_indices
            ), f"{num_train=} + {num_val=} + {num_test=} != {num_indices=}"

            # Dump the indices to root/metadata/split_indices.npz
            cls.save_indices(split_indices, root_path, dataset)

            def _dump_idx(indices: np.ndarray):
                nonlocal all_conformers, f

                # Dump the systems
                for idx in indices:
                    idx = int(idx)

                    # Get the molecule and conformer indices
                    mol_idx, conf_idx = all_conformers[idx]

                    # Get the molecule
                    mol = f[mol_idx]
                    assert isinstance(mol, h5py.Group), f"{mol=} is not a group"

                    atomic_numbers = atomic_numbers = torch.from_numpy(
                        np.array(_assert_dataset(mol.get("atomic_numbers")))
                    ).long()  # n_atoms
                    y = torch.from_numpy(
                        np.array(_assert_dataset(mol.get("dft_total_energy"))[conf_idx])
                    ).float()  # ()
                    formation_energy = torch.from_numpy(
                        np.array(_assert_dataset(mol.get("formation_energy"))[conf_idx])
                    ).float()  # ()
                    force = torch.from_numpy(
                        np.array(
                            _assert_dataset(mol.get("dft_total_gradient"))[conf_idx]
                        )
                    ).float()  # n_atoms 3
                    pos = torch.from_numpy(
                        np.array(_assert_dataset(mol.get("conformations"))[conf_idx])
                    ).float()  # n_atoms 3

                    data_object = Data(
                        atomic_numbers=atomic_numbers,
                        pos=pos,
                        force=force,
                        y=y,
                        formation_energy=formation_energy,
                        sid=f"{mol_idx}__{conf_idx}",
                    )
                    yield data_object

            # Convert the raw data to LMDB
            log.info("Converting raw data to LMDB")

            # Make the processed data directory
            lmdb_path = root_path / "lmdb" / dataset
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
                    / f"{dataset}.npz",
                )

    @override
    @classmethod
    def pre_data_transform(cls, data: Data) -> Data:
        data = super().pre_data_transform(data)
        data = update_units_transform(
            data, ["y", "force", "formation_energy"], from_="hartree", to="eV"
        )
        return data


if __name__ == "__main__":
    import argparse

    from tqdm.auto import tqdm

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(SPICE.datasets.keys())
    for dataset in pbar:
        pbar.set_description(dataset)
        SPICE.download(args.destination, dataset)
