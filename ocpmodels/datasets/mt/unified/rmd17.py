import os
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from typing_extensions import TypeVar, override

from ...modules.transforms.units import update_units_transform
from .base import LmdbDataset

log = getLogger(__name__)

T = TypeVar("T", infer_variance=True)

DOWNLOAD_URL = "https://figshare.com/ndownloader/articles/12672038/versions/3"
DOWNLOAD_FILENAME = "12672038.zip"
TRAIN_SIZE = 1000
VAL_SIZE = 50


class MD17(LmdbDataset[T]):
    name_mappings = {
        "napthalene": "naphthalene",
        "salicylic acid": "salicylic",
        "salicylic-acid": "salicylic",
        "salicylic_acid": "salicylic",
    }
    molecules = {
        "aspirin",
        "azobenzene",
        "benzene",
        "ethanol",
        "malonaldehyde",
        "naphthalene",
        "paracetamol",
        "salicylic",
        "toluene",
        "uracil",
    }

    @classmethod
    def download(
        cls,
        destination: str | Path,
        molecule: str,
        random_seed: int = 42,
    ):
        global DOWNLOAD_URL, DOWNLOAD_FILENAME, TRAIN_SIZE, VAL_SIZE
        molecule = cls.name_mappings.get(molecule, molecule)

        if molecule not in cls.molecules:
            raise ValueError(f"Invalid molecule: {molecule}")

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

            # Untar the rmd17 data
            log.info("Untarring rmd17 data")
            _ = os.system(f"tar -xjf {raw_dir / 'rmd17.tar.bz2'} -C {raw_dir}")
        else:
            log.info("Raw data already downloaded")

        # Load the raw data
        path = raw_dir / f"rmd17/npz_data/rmd17_{molecule}.npz"
        log.info(f"Loading raw data from {path}")
        data = np.load(path)

        atomic_numbers = torch.from_numpy(data["nuclear_charges"]).long()  # natoms
        pos = torch.from_numpy(data["coords"]).float()  # nframes natoms 3
        y = torch.from_numpy(data["energies"]).float()  # nframes
        force = torch.from_numpy(data["forces"]).float()  # nframes natoms 3

        n_frames, _, _ = pos.shape

        # Shuffle the frames
        frame_indices = np.arange(n_frames)
        np.random.RandomState(random_seed).shuffle(frame_indices)

        # Split into train/test based on train_size
        train_size = TRAIN_SIZE
        test_size = n_frames - train_size

        # Split train into train/val using 95/5 split
        val_size = VAL_SIZE
        train_size -= VAL_SIZE

        # Make sure the frames add up
        assert train_size + val_size + test_size == n_frames, "Frames don't add up"

        # Create train/val/test splits
        split_indices = {
            "train": frame_indices[:train_size],
            "val": frame_indices[train_size : train_size + val_size],
            "test": frame_indices[train_size + val_size :],
        }
        split_sizes = {
            name: indices.shape[0] for name, indices in split_indices.items()
        }
        log.critical(f"Created the following splits: {split_sizes}")

        # Dump the indices to root/metadata/split_indices.npz
        cls.save_indices(split_indices, root_path, f"{molecule}.npz")

        def _dump_frames(indices: np.ndarray):
            nonlocal atomic_numbers, pos, y, force

            # Dump the frames
            for frame_idx in indices:
                data_object = Data(
                    atomic_numbers=atomic_numbers.clone(),
                    pos=pos[frame_idx].clone(),
                    y=y[frame_idx].clone(),
                    force=force[frame_idx].clone(),
                    sid=torch.tensor(frame_idx, dtype=torch.long),
                )
                yield data_object

        # Make lmdb directory
        lmdb_path = root_path / "lmdb" / molecule
        lmdb_path.mkdir(parents=True, exist_ok=True)

        # Dump the frames
        log.info("Converting raw data to LMDB")
        for split, indices in split_indices.items():
            path = lmdb_path / split
            path.mkdir(parents=True, exist_ok=True)

            cls.dump_data(
                _dump_frames(indices),
                count=indices.shape[0],
                path=path,
                natoms_metadata_additional_path=root_path
                / "metadata"
                / split
                / f"{molecule}.npz",
            )

    @override
    @classmethod
    def pre_data_transform(cls, data: Data) -> Data:
        data = super().pre_data_transform(data)
        data = update_units_transform(data, ["y", "force"], from_="kcal/mol", to="eV")
        return data


if __name__ == "__main__":
    import argparse

    from tqdm.auto import tqdm

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    for molecule in tqdm(MD17.molecules):
        MD17.download(args.destination, molecule)
