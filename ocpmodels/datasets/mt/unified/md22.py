import os
from logging import getLogger
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch_geometric.data import Data
from typing_extensions import TypeVar, override

from ...modules.transforms import update_units_transform
from .base import LmdbDataset

log = getLogger(__name__)

T = TypeVar("T", infer_variance=True)


class MoleculeInfo(TypedDict):
    train_size: int


DOWNLOAD_ROOT = "http://www.quantum-machine.org/gdml/repo/datasets/"
VAL_SIZE = 0.05


class MD22(LmdbDataset[T]):
    molecules: dict[str, MoleculeInfo] = {
        "Ac-Ala3-NHMe": {"train_size": 6000},
        "DHA": {"train_size": 8000},
        "stachyose": {"train_size": 8000},
        "AT-AT": {"train_size": 3000},
        "AT-AT-CG-CG": {"train_size": 2000},
        "buckyball-catcher": {"train_size": 600},
        "double-walled_nanotube": {"train_size": 800},
    }

    @classmethod
    def download(
        cls,
        destination: str | Path,
        molecule: str,
        random_seed: int = 42,
    ):
        global DOWNLOAD_ROOT, VAL_SIZE

        if (info := cls.molecules.get(molecule)) is None:
            raise ValueError(f"{molecule=} is not a valid MD22 molecule name.")

        # Create root directory
        root_path = Path(destination)
        root_path.mkdir(parents=True, exist_ok=True)

        # Download dataset
        dl_path = root_path / "raw/"
        dl_path.mkdir(parents=True, exist_ok=True)
        npz_file = dl_path / f"md22_{molecule}.npz"
        if not npz_file.exists():
            log.info(f"Downloading {npz_file}")
            _ = os.system(f"wget -q {DOWNLOAD_ROOT}md22_{molecule}.npz -P {dl_path}")

        # Load data
        """
        NPZ file data for Ac-Ala3-NHMe:
            {'E': array[85109, 1] x∈[-6.207e+05, -6.206e+05] μ=-6.207e+05 σ=8.204,
            'E_max': array(-620623.81595481),
            'E_mean': array(-620662.71173186),
            'E_min': array(-620726.00266174),
            'E_var': array(67.30187507),
            'F': array[85109, 42, 3] n=10723734 x∈[-221.883, 216.102] μ=-1.702e-09 σ=26.039,
            'F_max': array(216.10170499),
            'F_mean': array(-1.70237435e-09),
            'F_min': array(-221.88345657),
            'F_var': array(678.01604927),
            'R': array[85109, 42, 3] n=10723734 x∈[-7.323, 7.873] μ=0.008 σ=2.187,
            'code_version': array('0.4.18.dev1', dtype='<U11'),
            'e_unit': array('kcal/mol', dtype='<U8'),
            'md5': array(b'566c78362b66a81c9cb56f9275dbfa0f', dtype='|S32'),
            'name': array('Ac-Ala3-NHMe', dtype='<U12'),
            'perms': array[18, 42] i64 n=756 x∈[0, 41] μ=20.500 σ=12.121,
            'r_unit': array('Ang', dtype='<U3'),
            'theory': array('PBE+MBD, 500K', dtype='<U13'),
            'type': array(b'd', dtype='|S1'),
            'z': array[42] i64 x∈[1, 8] μ=3.667 σ=2.851}
        """
        log.info(f"Loading {npz_file}")
        data = np.load(npz_file, allow_pickle=True)
        E = torch.from_numpy(data["E"]).float()  # F 1
        F = torch.from_numpy(data["F"]).float()  # F natoms 3
        pos = torch.from_numpy(data["R"]).float()  # F natoms 3
        atomic_numbers = torch.from_numpy(data["z"]).long()  # natoms

        n_frames, _, _ = pos.shape

        # Shuffle the frames
        all_indices = np.arange(n_frames)
        np.random.RandomState(random_seed).shuffle(all_indices)

        # Split into train/test based on train_size
        train_size = info["train_size"]
        test_size = n_frames - train_size

        # Split train into train/val using 95/5 split
        val_size = int(train_size * VAL_SIZE)
        train_size -= val_size

        # Make sure the frames add up
        assert train_size + val_size + test_size == n_frames, "Frames don't add up"

        # Create train/val/test splits
        split_indices = {
            "train": all_indices[:train_size],
            "val": all_indices[train_size : train_size + val_size],
            "test": all_indices[train_size + val_size :],
        }
        split_sizes = {
            name: indices.shape[0] for name, indices in split_indices.items()
        }
        log.critical(f"Created the following splits: {split_sizes}")

        # Dump the indices to root/metadata/split_indices.npz
        cls.save_indices(split_indices, root_path, f"{molecule}.npz")

        def _dump_frames(indices: np.ndarray):
            nonlocal atomic_numbers, pos, E, F

            # Dump the frames
            for frame_idx in indices:
                data_object = Data(
                    atomic_numbers=atomic_numbers.clone(),
                    pos=pos[frame_idx].clone(),
                    y=E[frame_idx].clone(),
                    force=F[frame_idx].clone(),
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

    pbar = tqdm(MD22.molecules.keys())
    for molecule in pbar:
        pbar.set_description(molecule)
        MD22.download(args.destination, molecule)
