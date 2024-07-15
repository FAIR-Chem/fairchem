import os
import numpy as np
import pytest

from fairchem.core.datasets import LMDBDatabase, create_dataset
from fairchem.core.datasets.base_dataset import BaseDataset
import tempfile


@pytest.fixture()
def lmdb_database(structures):
    with tempfile.TemporaryDirectory() as tmpdirname:
        num_atoms = []
        asedb_fn = f"{tmpdirname}/asedb.lmdb"
        with LMDBDatabase(asedb_fn) as database:
            for i, atoms in enumerate(structures):
                database.write(atoms, data=atoms.info)
                num_atoms.append(len(atoms))
        np.savez(f"{tmpdirname}/metadata.npz", natoms=num_atoms)
        yield asedb_fn


@pytest.mark.parametrize("max_atoms", [3, None])
@pytest.mark.parametrize(
    "key, value", [("first_n", 2), ("sample_n", 2), ("no_shuffle", True)]
)
def test_create_dataset(key, value, max_atoms, structures, lmdb_database):
    # now create a config
    config = {
        "format": "ase_db",
        "src": str(lmdb_database),
        key: value,
        "max_atoms": max_atoms,
    }

    dataset = create_dataset(config, split="train")
    if max_atoms is not None:
        structures = [s for s in structures if len(s) <= max_atoms]
        assert all(
            natoms <= max_atoms
            for natoms in dataset.metadata.natoms[range(len(dataset))]
        )
    if key == "first_n":  # this assumes first_n are not shuffled
        assert all(
            np.allclose(a1.cell.array, a2.cell.numpy())
            for a1, a2 in zip(structures[:value], dataset)
        )
        assert all(
            np.allclose(a1.numbers, a2.atomic_numbers)
            for a1, a2 in zip(structures[:value], dataset)
        )
    elif key == "sample_n":
        assert len(dataset) == value
    else:  # no shuffle all of them are in there
        assert all(
            np.allclose(a1.cell.array, a2.cell.numpy())
            for a1, a2 in zip(structures, dataset)
        )
        assert all(
            np.allclose(a1.numbers, a2.atomic_numbers)
            for a1, a2 in zip(structures, dataset)
        )


# make sure we cant sample more than the number of elements in the dataset with sample_n
def test_sample_n_dataset(lmdb_database):
    with pytest.raises(ValueError):
        _ = create_dataset(
            config={
                "format": "ase_db",
                "src": str(lmdb_database),
                "sample_n": 100,
            },
            split="train",
        )


def test_diff_seed_sample_dataset(lmdb_database):
    dataset_a = create_dataset(
        config={
            "format": "ase_db",
            "src": str(lmdb_database),
            "sample_n": 3,
            "seed": 0,
        },
        split="train",
    )
    dataset_b = create_dataset(
        config={
            "format": "ase_db",
            "src": str(lmdb_database),
            "sample_n": 3,
            "seed": 0,
        },
        split="train",
    )
    assert (dataset_a.indices == dataset_b.indices).all()
    dataset_b = create_dataset(
        config={
            "format": "ase_db",
            "src": str(lmdb_database),
            "sample_n": 3,
            "seed": 1,
        },
        split="train",
    )
    assert not (dataset_a.indices == dataset_b.indices).all()


def test_del_dataset():
    class _Dataset(BaseDataset):
        def __init__(self, fn) -> None:
            super().__init__(config={})
            self.fn = fn
            open(self.fn, "a").close()

        def __del__(self):
            os.remove(self.fn)

    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = tmpdirname + "/test"
        d = _Dataset(fn)
        del d
        assert not os.path.exists(fn)
