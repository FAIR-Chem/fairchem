import numpy as np
import pytest

from ocpmodels.datasets import LMDBDatabase, create_dataset


@pytest.mark.parametrize("max_atoms", [3, None])
@pytest.mark.parametrize(
    "key, value", [("first_n", 2), ("sample_n", 2), ("no_shuffle", True)]
)
def test_create_dataset(key, value, max_atoms, structures, tmp_path):
    # write a dataset and metadata file
    num_atoms = []
    with LMDBDatabase(str(tmp_path / "asedb.lmdb")) as database:
        for i, atoms in enumerate(structures):
            database.write(atoms, data=atoms.info)
            num_atoms.append(len(atoms))
    np.savez(tmp_path / "metadata.npz", natoms=num_atoms)
    print(num_atoms)
    # now create a config
    config = {
        "format": "ase_db",
        "src": str(tmp_path / "asedb.lmdb"),
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
