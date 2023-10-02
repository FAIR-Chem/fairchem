import os

import numpy as np
from ase import build, db
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory, write

from ocpmodels.datasets import (
    AseDBDataset,
    AseReadDataset,
    AseReadMultiStructureDataset,
)
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


def test_ase_read_dataset() -> None:
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

    dataset.close_db()

    for i in range(len(structures)):
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), f"{i}.cif"
            )
        )


def test_ase_db_dataset() -> None:
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


def test_ase_db_dataset_folder() -> None:
    try:
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb1.db"
            )
        )
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb2.db"
            )
        )
    except FileNotFoundError:
        pass

    with db.connect(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb1.db")
    ) as database:
        for i, structure in enumerate(structures):
            database.write(structure)

    with db.connect(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb2.db")
    ) as database:
        for i, structure in enumerate(structures):
            database.write(structure)

    dataset = AseDBDataset(
        config={
            "src": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "./"
            ),
        }
    )

    assert len(dataset) == len(structures) * 2
    data = dataset[0]
    del data

    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb1.db")
    )
    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb2.db")
    )


def test_ase_db_dataset_list() -> None:
    try:
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb1.db"
            )
        )
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb2.db"
            )
        )
    except FileNotFoundError:
        pass

    with db.connect(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb1.db")
    ) as database:
        for i, structure in enumerate(structures):
            database.write(structure)

    with db.connect(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb2.db")
    ) as database:
        for i, structure in enumerate(structures):
            database.write(structure)

    dataset = AseDBDataset(
        config={
            "src": [
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "asedb1.db"
                ),
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "asedb2.db"
                ),
            ]
        }
    )

    assert len(dataset) == len(structures) * 2
    data = dataset[0]
    del data

    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb1.db")
    )
    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb2.db")
    )


def test_ase_lmdb_dataset() -> None:
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


def test_lmdb_metadata_guesser() -> None:
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


def test_ase_metadata_guesser() -> None:
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

    dataset = AseDBDataset(
        config={
            "src": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.db"
            ),
        }
    )

    database.delete([1])

    new_structures = [
        build.molecule("CH3COOH", vacuum=4),
        build.bulk("Al"),
    ]

    for i, structure in enumerate(new_structures):
        database.write(structure)

    dataset = AseDBDataset(
        config={
            "src": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.db"
            ),
        }
    )

    assert len(dataset) == len(structures) + len(new_structures) - 1
    data = dataset[:]
    assert data

    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db")
    )

    dataset.close_db()


def test_ase_multiread_dataset() -> None:
    try:
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test.traj"
            )
        )
    except FileNotFoundError:
        pass

    try:
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test_index_file"
            )
        )
    except FileNotFoundError:
        pass

    atoms_objects = [build.bulk("Cu", a=a) for a in np.linspace(3.5, 3.7, 10)]

    energies = np.linspace(1, 0, len(atoms_objects))

    traj = Trajectory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.traj"),
        mode="w",
    )

    for atoms, energy in zip(atoms_objects, energies):
        calc = SinglePointCalculator(
            atoms, energy=energy, forces=atoms.positions
        )
        atoms.calc = calc
        traj.write(atoms)

    dataset = AseReadMultiStructureDataset(
        config={
            "src": os.path.join(os.path.dirname(os.path.abspath(__file__))),
            "pattern": "*.traj",
            "keep_in_memory": True,
            "atoms_transform_args": {
                "skip_always": True,
            },
        }
    )

    assert len(dataset) == len(atoms_objects)
    [dataset[:]]

    f = open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_index_file"
        ),
        "w",
    )
    f.write(
        f"{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.traj')} {len(atoms_objects)}"
    )
    f.close()

    dataset = AseReadMultiStructureDataset(
        config={
            "index_file": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test_index_file"
            )
        },
    )

    assert len(dataset) == len(atoms_objects)
    [dataset[:]]

    dataset = AseReadMultiStructureDataset(
        config={
            "index_file": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test_index_file"
            ),
            "a2g_args": {
                "r_energy": True,
                "r_forces": True,
            },
            "include_relaxed_energy": True,
        }
    )

    assert len(dataset) == len(atoms_objects)
    [dataset[:]]

    assert hasattr(dataset[0], "y_relaxed")
    assert dataset[0].y_relaxed != dataset[0].y
    assert dataset[-1].y_relaxed == dataset[-1].y

    dataset = AseReadDataset(
        config={
            "src": os.path.join(os.path.dirname(os.path.abspath(__file__))),
            "pattern": "*.traj",
            "ase_read_args": {
                "index": "0",
            },
            "a2g_args": {
                "r_energy": True,
                "r_forces": True,
            },
            "include_relaxed_energy": True,
        }
    )

    [dataset[:]]

    assert hasattr(dataset[0], "y_relaxed")
    assert dataset[0].y_relaxed != dataset[0].y

    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.traj")
    )
    os.remove(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_index_file"
        )
    )
