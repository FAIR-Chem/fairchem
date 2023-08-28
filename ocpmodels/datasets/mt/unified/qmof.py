import json
import os
from logging import getLogger
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
from torch_geometric.data import Data
from typing_extensions import TypeVar

from .base import LmdbDataset
from .utils import atomic_symbol_to_element

log = getLogger(__name__)

T = TypeVar("T", infer_variance=True)


# region Types
# region Structure
class Lattice(TypedDict):
    matrix: list[list[float]]
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    volume: float


class Properties(TypedDict):
    pbe_ddec_sum_bond_order: float
    pbe_ddec_charge: float
    pbe_cm5_charge: float
    pbe_bader_charge: float
    pbe_magmom: float
    pbe_ddec_spin_density: float
    pbe_bader_spin_density: float


class Species(TypedDict):
    element: str
    occu: int


class Site(TypedDict):
    species: list[Species]
    abc: list[float]
    xyz: list[float]
    label: str
    properties: Properties


class Structure(TypedDict):
    module: str
    structure_class: str
    charge: int
    lattice: Lattice
    sites: list[Site]


class QMOFStructureDict(TypedDict):
    qmof_id: str
    name: str
    structure: Structure


# endregion

# region System


class Mofid(TypedDict):
    mofid: None
    mofkey: None
    smiles_nodes: list[str]
    smiles_linkers: list[str]
    smiles: str
    topology: None


class Symmetry(TypedDict):
    spacegroup: str
    spacegroupnumber: int
    spacegroupcrystal: str
    pointgroup: int


class Info(TypedDict):
    formula: str
    formula_reduced: str
    mofid: Mofid
    natoms: int
    pld: float
    lcd: float
    density: float
    volume: float
    symmetry: Symmetry
    synthesized: bool
    source: str
    doi: str


class InputsPbe(TypedDict):
    theory: str
    pseudopotentials: list[str]
    encut: int
    kpoints: list[int]
    gamma: bool
    spin: bool


class Inputs(TypedDict):
    pbe: InputsPbe


class OutputsPbe(TypedDict):
    energy_total: float
    energy_vdw: float
    energy_elec: float
    net_magmom: int
    bandgap: float
    cbm: float
    vbm: float
    directgap: bool
    bandgap_spins: list[float]
    cbm_spins: list[float]
    vbm_spins: list[float]
    directgap_spins: list[bool]


class Outputs(TypedDict):
    pbe: OutputsPbe


class QMOFSystemDict(TypedDict):
    qmof_id: str
    name: str
    info: Info
    inputs: Inputs
    outputs: Outputs


# endregion
# endregion

DOWNLOAD_URL = "https://figshare.com/ndownloader/articles/13147324/versions/14"
DOWNLOAD_FILENAME = "13147324.zip"


class QMOF(LmdbDataset[T]):
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

            # Unzip the "qmof_database.zip" file
            log.info("Unzipping qmof_database.zip")
            _ = os.system(f"unzip {raw_dir / 'qmof_database.zip'} -d {raw_dir}")
        else:
            log.info("Raw data already downloaded")

        # Load the raw data
        log.info("Loading raw data")
        with open(raw_dir / "qmof_database" / "qmof.json") as f:
            qmof_data = json.load(f)

        with open(raw_dir / "qmof_database" / "qmof_structure_data.json") as f:
            qmof_structure_data = json.load(f)

        # Both of these are lists of dicts and should be the same length
        assert isinstance(qmof_data, list), f"{type(qmof_data)=} is not list"
        assert isinstance(
            qmof_structure_data, list
        ), f"{type(qmof_structure_data)=} is not list"
        assert len(qmof_data) == len(
            qmof_structure_data
        ), f"{len(qmof_data)=} != {len(qmof_structure_data)=}"
        qmof_data = cast(list[QMOFSystemDict], qmof_data)
        qmof_structure_data = cast(list[QMOFStructureDict], qmof_structure_data)

        # Get the indices for each split (80/10/10 train/val/test)
        all_indices = np.arange(len(qmof_data))
        np.random.RandomState(random_seed).shuffle(all_indices)
        num_indices = len(all_indices)
        num_train = int(num_indices * 0.8)
        num_test = int(num_indices * 0.1)
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
        cls.save_indices(split_indices, root_path)

        def _dump_idx(indices: np.ndarray):
            nonlocal qmof_data, qmof_structure_data

            # Dump the systems
            for idx in indices:
                idx = int(idx)
                # Get the system and structure data
                system = qmof_data[idx]
                structure = qmof_structure_data[idx]

                atomic_numbers = torch.tensor(
                    [
                        atomic_symbol_to_element[site["label"]]
                        for site in structure["structure"]["sites"]
                    ],
                    dtype=torch.long,
                )  # natoms
                pos = torch.tensor(
                    [site["xyz"] for site in structure["structure"]["sites"]],
                    dtype=torch.float,
                )  # natoms 3
                cell = torch.tensor(
                    structure["structure"]["lattice"]["matrix"],
                    dtype=torch.float,
                ).unsqueeze(
                    dim=0
                )  # 1 3 3
                band_gap = torch.tensor(
                    system["outputs"]["pbe"]["bandgap"],
                    dtype=torch.float,
                )  # ()
                energy_total = torch.tensor(
                    system["outputs"]["pbe"]["energy_total"],
                    dtype=torch.float,
                )  # ()

                data_object = Data(
                    atomic_numbers=atomic_numbers,
                    pos=pos,
                    cell=cell,
                    y=band_gap,
                    energy_total=energy_total,
                    sid=system["qmof_id"],
                )
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

    QMOF.download(args.destination)
