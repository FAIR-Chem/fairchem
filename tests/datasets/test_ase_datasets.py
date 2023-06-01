import pytest
from ase import build, db
from ase.io import write, Trajectory
import os
import numpy as np

from ocpmodels.datasets import (
    AseReadDataset,
    AseDBDataset,
    AseReadMultiStructureDataset,
)

structures = [
    build.molecule("H2O", vacuum=4),
    build.bulk("Cu"),
    build.fcc111("Pt", size=[2, 2, 3], vacuum=8, periodic=True),
]


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

    for i in range(len(structures)):
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), f"{i}.cif"
            )
        )

    dataset.close_db()


def test_ase_db_dataset():
    try:
        os.remove(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "asedb.db"
            )
        )
    except FileNotFoundError:
        pass

    database = db.connect(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db")
    )
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

    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db")
    )

    dataset.close_db()


def test_ase_multiread_dataset():
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

    traj = Trajectory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.traj"),
        mode="w",
    )
    for atoms in atoms_objects:
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

    os.remove(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.traj")
    )
    os.remove(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_index_file"
        )
    )
