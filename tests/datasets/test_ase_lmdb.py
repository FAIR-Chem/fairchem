from pathlib import Path

import numpy as np
import tqdm
from ase import build
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from ocpmodels.datasets.lmdb_database import LMDBDatabase

DB_NAME = "ase_lmdb.lmdb"
N_WRITES = 100
N_READS = 200


def cleanup_asedb() -> None:
    if Path(DB_NAME).is_file():
        Path(DB_NAME).unlink()
    if Path(f"{DB_NAME}-lock").is_file():
        Path(f"{DB_NAME}-lock").unlink()


test_structures = [
    build.molecule("H2O", vacuum=4),
    build.bulk("Cu"),
    build.fcc111("Pt", size=[2, 2, 3], vacuum=8, periodic=True),
]

test_structures[2].set_constraint(FixAtoms(indices=[0, 1]))


def generate_random_structure():
    # Make base slab
    slab = build.fcc111("Cu", size=(4, 4, 3), vacuum=10.0)

    # Randomly set some elements
    slab.set_chemical_symbols(
        np.random.choice(["Cu", "Ag", "Au", "Pd"], size=(len(slab)))
    )

    # Randomly set some positions
    slab.positions = np.random.normal(size=slab.positions.shape)

    # Add entries for energy/forces/stress/magmom/etc.
    # Property must be one of the ASE core properties to
    # go in to a singlepointcalculator and get stored as
    # fields correctly
    spc = SinglePointCalculator(
        slab,
        energy=np.random.normal(),
        forces=np.random.normal(size=slab.positions.shape),
        stress=np.random.normal(size=(3, 3)),
        magmom=np.random.normal(size=(len(slab))),
    )
    slab.set_calculator(spc)

    # Make up some other properties to show how we can include arbitrary outputs
    slab.info["test_info_property_1"] = np.random.normal(size=(3, 3))
    slab.info["test_info_property_2"] = np.random.normal(size=(len(slab), 3))

    return slab


def write_random_atoms() -> None:
    slab = build.fcc111("Cu", size=(4, 4, 3), vacuum=10.0)
    with LMDBDatabase(DB_NAME) as db:
        for structure in test_structures:
            db.write(structure)

        for i in tqdm.tqdm(range(N_WRITES)):
            slab = generate_random_structure()

            # Save the slab info, and make sure the info gets put in as data
            db.write(slab, data=slab.info)


def test_aselmdb_write() -> None:
    # Representative structure
    write_random_atoms()

    with LMDBDatabase(DB_NAME, readonly=True) as db:
        for i, structure in enumerate(test_structures):
            assert str(structure) == str(db._get_row_by_index(i).toatoms())

    cleanup_asedb()


def test_aselmdb_count() -> None:
    # Representative structure
    write_random_atoms()

    with LMDBDatabase(DB_NAME, readonly=True) as db:
        assert db.count() == N_WRITES + len(test_structures)

    cleanup_asedb()


def test_aselmdb_delete() -> None:
    cleanup_asedb()

    # Representative structure
    write_random_atoms()

    with LMDBDatabase(DB_NAME) as db:
        for i in range(5):
            # Note the available ids list is updating
            # but the ids themselves are fixed.
            db.delete([db.ids[0]])

    assert db.count() == N_WRITES + len(test_structures) - 5

    cleanup_asedb()


def test_aselmdb_randomreads() -> None:
    write_random_atoms()

    with LMDBDatabase(DB_NAME, readonly=True) as db:
        for i in tqdm.tqdm(range(N_READS)):
            total_size = db.count()
            row = db._get_row_by_index(np.random.choice(total_size)).toatoms()
            del row
    cleanup_asedb()


def test_aselmdb_constraintread() -> None:
    write_random_atoms()

    with LMDBDatabase(DB_NAME, readonly=True) as db:
        atoms = db._get_row_by_index(2).toatoms()

    assert type(atoms.constraints[0]) == FixAtoms

    cleanup_asedb()


def update_keyvalue_pair() -> None:
    write_random_atoms()
    with LMDBDatabase(DB_NAME) as db:
        db.update(1, test=5)

    with LMDBDatabase(DB_NAME) as db:
        row = db.get_row_by_id(1)
        assert row.test == 5

    cleanup_asedb()


def update_atoms() -> None:
    write_random_atoms()
    with LMDBDatabase(DB_NAME) as db:
        db.update(40, atoms=test_structures[-1])

    with LMDBDatabase(DB_NAME) as db:
        row = db.get_row_by_id(40)
        assert str(row.toatoms()) == str(test_structures[-1])

    cleanup_asedb()


def test_metadata() -> None:
    write_random_atoms()

    with LMDBDatabase(DB_NAME) as db:
        db.metadata = {"test": True}

    with LMDBDatabase(DB_NAME, readonly=True) as db:
        assert db.metadata["test"] is True

    cleanup_asedb()
