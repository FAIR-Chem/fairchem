from logging import getLogger
from pathlib import Path
from typing import cast

import numpy as np
import torch
from matbench.bench import MatbenchBenchmark, MatbenchTask
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from typing_extensions import TypeVar

from .base import LmdbDataset
from .utils import env

log = getLogger(__name__)


T = TypeVar("T", infer_variance=True)


class MatBench(LmdbDataset[T]):
    tasks = [
        "matbench_jdft2d",
        "matbench_phonons",
        "matbench_dielectric",
        "matbench_log_gvrh",
        "matbench_log_kvrh",
        "matbench_mp_is_metal",
        "matbench_perovskites",
        "matbench_mp_gap",
        "matbench_mp_e_form",
    ]
    folds = [0, 1, 2, 3, 4]

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

        def _dump_split(data: tuple[list[Structure], list[float]], indices: np.ndarray):
            inputs, outputs = data
            # Make sure the sizes match
            assert len(inputs) == len(outputs), f"{len(inputs)=} != {len(outputs)=}"

            # Dump the systems
            for idx in indices:
                idx = int(idx)
                # Get the system and structure data
                structure = inputs[idx]
                output = outputs[idx]

                atomic_numbers = torch.tensor(
                    [site.specie.number for site in structure], dtype=torch.long
                )  # natoms
                cell = torch.tensor(
                    structure.lattice.matrix, dtype=torch.float
                ).unsqueeze(
                    dim=0
                )  # 1 3 3
                pos = torch.tensor(
                    [site.coords for site in structure], dtype=torch.float
                )  # natoms 3
                y = torch.tensor(output)  # ()
                if isinstance(output, bool):
                    y = y.bool()
                else:
                    y = y.float()

                data_object = Data(
                    atomic_numbers=atomic_numbers,
                    pos=pos,
                    cell=cell,
                    y=y,
                )
                yield data_object

        with env({"MATMINER_DATA": str(raw_dir.absolute())}):
            mb = MatbenchBenchmark(autoload=False, subset=cls.tasks)
            mb.load()
            for task in mb.tasks:
                task = cast(MatbenchTask, task)
                for fold in cls.folds:
                    train_val_data = task.get_train_and_val_data(fold, as_type="tuple")
                    assert isinstance(
                        train_val_data, tuple
                    ), f"{type(train_val_data)=} is not tuple"
                    test_data = task.get_test_data(
                        fold, as_type="tuple", include_target=True
                    )
                    assert isinstance(
                        test_data, tuple
                    ), f"{type(test_data)=} is not tuple"

                    # Get the train/val indices
                    num_train_val_indices = len(train_val_data[0])
                    all_indices = np.arange(num_train_val_indices)
                    np.random.RandomState(random_seed).shuffle(all_indices)
                    num_indices = len(all_indices)
                    num_train = int(num_indices * 0.9)
                    num_val = num_indices - num_train
                    split_indices = {
                        "train": all_indices[:num_train],
                        "val": all_indices[num_train:],
                    }
                    # Test is a separate dataset, so we don't need to split it
                    test_indices = np.arange(len(test_data[0]))
                    split_indices["test"] = test_indices

                    # Make sure the splits add up
                    assert (
                        num_train + num_val == num_train_val_indices
                    ), f"{num_train=} + {num_val=} != {num_train_val_indices=}"

                    # Dump the indices to root/metadata/split_indices.npz
                    cls.save_indices(
                        split_indices,
                        root_path,
                        f"{task.dataset_name}_{fold}.npz",
                    )

                    # Convert the raw data to LMDB
                    log.info("Converting raw data to LMDB")

                    # Make the processed data directory
                    lmdb_path = root_path / "lmdb" / task.dataset_name / str(fold)
                    lmdb_path.mkdir(parents=True, exist_ok=True)

                    # Dump the frames
                    for split, indices in split_indices.items():
                        path = lmdb_path / split
                        path.mkdir(parents=True, exist_ok=True)

                        data = train_val_data if split != "test" else test_data

                        cls.dump_data(
                            _dump_split(data, indices),
                            count=indices.shape[0],
                            path=path,
                            natoms_metadata_additional_path=root_path
                            / "metadata"
                            / split
                            / f"{task.dataset_name}_{fold}.npz",
                        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    _ = parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    MatBench.download(destination=args.destination)
