import os

import pytest
from ase import build, db
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from ocpmodels.datasets import AseDBDataset, AseReadDataset
from ocpmodels.datasets.lmdb_database import LMDBDatabase

structures = [
    build.molecule("H2O", vacuum=4),
    build.bulk("Cu"),
    build.fcc111("Pt", size=[2, 2, 3], vacuum=8, periodic=True),
]
for atoms in structures:
    calc = SinglePointCalculator(atoms, energy=1, forces=atoms.positions)
    atoms.calc = calc
    atoms.info["test_extensive_property"] = 3 * len(atoms)

structures[2].set_pbc(True)


def test_ase_read_dataset():
    for i, structure in enumerate(structures):
        write(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), f"{i}.cif"
            ),
            structure,
        )

    dataset = AseReadDataset(
        config={
            "src": os.path.join(os.path.dirname(os.path.abspath(__file__))),
            "pattern": "*.cif",
        }
    )

    assert len(dataset) == len(structures)
    data = dataset[0]
    del data

    for i in range(len(structures)):
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), f"{i}.cif"
            )
        )


def test_ase_db_dataset():
    try:
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.db"
            )
        )
    except FileNotFoundError:
        pass

    with db.connect(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db")
    ) as database:
        for i, structure in enumerate(structures):
            database.write(structure)

    dataset = AseDBDataset(
        config={
            "src": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.db"
            ),
        }
    )

    assert len(dataset) == len(structures)
    data = dataset[0]
    del data

    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db")
    )


def test_ase_lmdb_dataset():
    try:
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.lmdb"
            )
        )
    except FileNotFoundError:
        pass

    with LMDBDatabase(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.lmdb")
    ) as database:
        for i, structure in enumerate(structures):
            database.write(structure)

    dataset = AseDBDataset(
        config={
            "src": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.lmdb"
            ),
        }
    )

    assert len(dataset) == len(structures)
    data = dataset[0]
    del data

    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.lmdb")
    )


def test_lmdb_metadata_guesser():

    # Cleanup old lmdb in case it's left over from previous tests
    try:
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.lmdb"
            )
        )
    except FileNotFoundError:
        pass

    # Write an LMDB
    with LMDBDatabase(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.lmdb")
    ) as database:
        for i, structure in enumerate(structures):
            database.write(structure, data=structure.info)

    dataset = AseDBDataset(
        config={
            "src": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.lmdb"
            ),
        }
    )

    metadata = dataset.get_metadata()

    # Confirm energy metadata guessed properly
    assert metadata["targets"]["energy"]["extensive"] is False
    assert metadata["targets"]["energy"]["shape"] == ()
    assert metadata["targets"]["energy"]["type"] == "per-image"

    # Confirm forces metadata guessed properly
    assert metadata["targets"]["forces"]["shape"] == (3,)
    assert metadata["targets"]["forces"]["extensive"] is True
    assert metadata["targets"]["forces"]["type"] == "per-atom"

    # Confirm forces metadata guessed properly
    assert (
        metadata["targets"]["info.test_extensive_property"]["extensive"]
        is True
    )
    assert metadata["targets"]["info.test_extensive_property"]["shape"] == ()
    assert (
        metadata["targets"]["info.test_extensive_property"]["type"]
        == "per-image"
    )

    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.lmdb")
    )


def test_ase_metadata_guesser():

    try:
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.db"
            )
        )
    except FileNotFoundError:
        pass

    with db.connect(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db")
    ) as database:
        for i, structure in enumerate(structures):
            database.write(structure, data=structure.info)

    dataset = AseDBDataset(
        config={
            "src": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.db"
            ),
        }
    )

    metadata = dataset.get_metadata()

    # Confirm energy metadata guessed properly
    assert metadata["targets"]["energy"]["extensive"] is False
    assert metadata["targets"]["energy"]["shape"] == ()
    assert metadata["targets"]["energy"]["type"] == "per-image"

    # Confirm forces metadata guessed properly
    assert metadata["targets"]["forces"]["shape"] == (3,)
    assert metadata["targets"]["forces"]["extensive"] is True
    assert metadata["targets"]["forces"]["type"] == "per-atom"

    # Confirm forces metadata guessed properly
    assert (
        metadata["targets"]["info.test_extensive_property"]["extensive"]
        is True
    )
    assert metadata["targets"]["info.test_extensive_property"]["shape"] == ()
    assert (
        metadata["targets"]["info.test_extensive_property"]["type"]
        == "per-image"
    )

    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db")
    )
