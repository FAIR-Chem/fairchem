import bisect
import pickle
from contextlib import ContextDecorator
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, Generator, TypedDict, cast

import lmdb
import numpy as np
import torch
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing_extensions import TypeVar, override

from ocpmodels.common.registry import registry

log = getLogger(__name__)


class Config(TypedDict):
    src: str | Path


def _pyg2_data_transform(data: Data):
    """
    if we're on the new pyg (2.0 or later) and if the Data stored is in older format
    we need to convert the data to the new format
    """
    if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
        return Data(
            **{k: v for k, v in data.__dict__.items() if v is not None}
        )

    return data


T = TypeVar("T", infer_variance=True)


@registry.register_dataset("mt_lmdb_unified")
class LmdbDataset(Dataset[T], ContextDecorator):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
    The keys in the LMDB must be integers (stored as ascii objects) starting
    from 0 through the length of the LMDB. For historical reasons any key named
    "length" is ignored since that was used to infer length of many lmdbs in the same
    folder, but lmdb lengths are now calculated directly from the number of keys.
    Args:
            config (dict): Dataset configuration
    """

    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.metadata["natoms"][indices]

    @cached_property
    def metadata(self) -> dict[str, np.ndarray]:
        metadata_path = self.metadata_path
        if metadata_path and metadata_path.is_file():
            return np.load(metadata_path, allow_pickle=True)

        raise ValueError(
            f"Could not find atoms metadata in '{self.metadata_path}'"
        )

    def data_transform(self, data: Data) -> Data:
        return data

    def __init__(
        self,
        src: str | Path,
        metadata_path: str | Path | None = None,
    ) -> None:
        super().__init__()

        self.path = Path(src)
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
        else:
            assert (
                self.path.suffix == ".lmdb"
            ), f"File '{self.path}' is not an LMDB"
            db_paths = [self.path]

        self.metadata_path = (
            Path(metadata_path)
            if metadata_path
            else self.path / "metadata.npz"
        )

        self.keys: list[list[int]] = []
        self.envs: list[lmdb.Environment] = []
        # Open all the lmdb files
        for db_path in db_paths:
            cur_env = lmdb.open(
                str(db_path.absolute()),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=1,
            )
            self.envs.append(cur_env)

            # If "length" encoded as ascii is present, use that
            length_entry = cur_env.begin().get("length".encode("ascii"))
            if length_entry is not None:
                num_entries = pickle.loads(length_entry)
            else:
                # Get the number of stores data from the number of entries
                # in the LMDB
                num_entries = cur_env.stat()["entries"]

            # Append the keys (0->num_entries) as a list
            self.keys.append(list(range(num_entries)))

        keylens = [len(k) for k in self.keys]
        self.keylen_cumulative: list[int] = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

    def __len__(self) -> int:
        return self.num_samples

    @override
    def __getitem__(self, idx: int):
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self.keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self.keylen_cumulative[db_idx - 1]
        assert el_idx >= 0, f"{el_idx=} is not a valid index"

        # Return features.
        key = f"{self.keys[db_idx][el_idx]}".encode("ascii")
        env = self.envs[db_idx]
        data_object_pickled = env.begin().get(key, default=None)
        if data_object_pickled is None:
            raise KeyError(
                f"Key {key=} not found in {env=}. {el_idx=} {db_idx=}"
            )

        data_object = _pyg2_data_transform(
            pickle.loads(cast(Any, data_object_pickled))
        )
        data_object.id = f"{db_idx}_{el_idx}"
        return data_object

    def close_db(self) -> None:
        for env in self.envs:
            env.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close_db()

    @classmethod
    def pre_data_transform(cls, data: Data) -> Data:
        if not hasattr(data, "tags"):
            data.tags = torch.full_like(data.atomic_numbers, 2)
        if not hasattr(data, "natoms"):
            data.natoms = data.num_nodes
        return data

    @classmethod
    def save_indices(
        cls,
        split_indices: dict[str, np.ndarray],
        root_path: Path,
        file_name: str | None = None,
    ):
        # Dump the indices to root/metadata/split_indices.npz
        if file_name is None:
            split_indices_path = root_path / "metadata"
            split_indices_path.mkdir(parents=True, exist_ok=True)
            split_indices_path = split_indices_path / "split_indices.npz"
        else:
            split_indices_path = root_path / "metadata/split_indices"
            split_indices_path.mkdir(parents=True, exist_ok=True)
            split_indices_path = split_indices_path / file_name

        np.savez(split_indices_path, **split_indices)
        split_indices_sizes = {
            name: indices.shape[0] for name, indices in split_indices.items()
        }
        log.critical(
            f"Dumped split_indices={split_indices_sizes} to {split_indices_path}"
        )

    @classmethod
    def dump_data(
        cls,
        generator: Generator[Data, None, None],
        count: int,
        path: Path,
        natoms_metadata_additional_path: Path | None = None,
        num_per_file: int = 5_000,
    ):
        natoms_metadata_list: list[int] = []
        # Store each chunk as "data.%04d.lmdb"
        for chunk_idx in range((count + num_per_file - 1) // num_per_file):
            # Create a new lmdb file
            chunk_path = path / f"data.{chunk_idx:04d}.lmdb"
            cur_env = lmdb.open(
                str(chunk_path.absolute()),
                map_size=1099511627776 * 2,
                subdir=False,
                meminit=False,
                map_async=True,
            )

            num_saved = 0
            with cur_env.begin(write=True) as txn:
                # Save the number of entries in this lmdb
                length = min(num_per_file, count - chunk_idx * num_per_file)
                txn.put("length".encode("ascii"), pickle.dumps(length))

                # Save the data
                for data_idx in range(min(num_per_file, length)):
                    data = next(generator)
                    data = cls.pre_data_transform(data)

                    # Get the natoms and save it for metadata
                    natoms_metadata_list.append(data.atomic_numbers.shape[0])

                    # Save the data
                    txn.put(f"{data_idx}".encode("ascii"), pickle.dumps(data))
                    num_saved += 1

            # Close the lmdb
            cur_env.close()
            log.critical(f"Saved {num_saved} entries to {chunk_path}")

        # Save the metadata
        natoms_metadata = np.array(natoms_metadata_list)
        for p in (natoms_metadata_additional_path, path / "metadata.npz"):
            if p is None:
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, natoms=natoms_metadata)
            log.critical(f"Saved metadata to {p}")
