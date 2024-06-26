from itertools import product
from random import choice
import pytest
import numpy.testing as npt
import numpy as np
import torch

from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure

from fairchem.core.datasets import LMDBDatabase, AseDBDataset
from fairchem.core.modules.element_references import (
    fit_linear_references,
    create_element_references,
    LinearReference,
)

# create some dummy elemental energies from ionic radii (ignore deuterium and tritium included in pmg)
DUMMY_EREFS = np.concatenate(
    [[0], [e.average_ionic_radius for e in Element if e.name not in ("D", "T")]]
)
MAX_NUM_ELEMENTS = len(DUMMY_EREFS) - 1


@pytest.fixture
def dataset(tmpdir):
    # a dummy dataset with binaries with energy that depends on composition only plus noise
    all_binaries = list(product(list(Element), repeat=2))
    rng = np.random.default_rng(seed=0)

    with LMDBDatabase(tmpdir / "dummy.aselmdb") as db:
        for _ in range(1000):
            elements = choice(all_binaries)
            structure = Structure.from_prototype("cscl", species=elements, a=2.0)
            energy = (
                sum(e.average_ionic_radius for e in elements)
                + 0.05 * rng.random() * DUMMY_EREFS.mean()
            )
            atoms = structure.to_ase_atoms()
            db.write(atoms, data={"energy": energy})

    dataset = AseDBDataset(
        config={
            "src": str(tmpdir / "dummy.aselmdb"),
            "a2g_args": {"r_data_keys": ["energy"]},
        }
    )
    return dataset


@pytest.fixture
def element_refs(dataset):
    return fit_linear_references(
        ["energy"],
        dataset,
        batch_size=16,
        shuffle=False,
        max_num_elements=MAX_NUM_ELEMENTS,
    )


def test_create_element_references(element_refs, tmpdir):
    # test from state dict
    sdict = element_refs["energy"].state_dict()

    refs = create_element_references(state_dict=sdict)
    assert isinstance(refs, LinearReference)
    assert torch.allclose(
        element_refs["energy"].element_references, refs.element_references
    )

    # test from saved stated dict
    torch.save(sdict, tmpdir / "linref.pt")
    refs = create_element_references(file=tmpdir / "linref.pt")
    assert isinstance(refs, LinearReference)
    assert torch.allclose(
        element_refs["energy"].element_references, refs.element_references
    )

    # from a legacy numpy npz file
    np.savez(
        tmpdir / "linref.npz", coeff=element_refs["energy"].element_references.numpy()
    )
    # breakpoint()
    refs = create_element_references(file=tmpdir / "linref.npz")
    assert isinstance(refs, LinearReference)
    assert torch.allclose(
        element_refs["energy"].element_references, refs.element_references
    )

    # from a numpy npz file
    np.savez(
        tmpdir / "linref.npz",
        element_references=element_refs["energy"].element_references.numpy(),
    )
    # breakpoint()
    refs = create_element_references(file=tmpdir / "linref.npz")
    assert isinstance(refs, LinearReference)
    assert torch.allclose(
        element_refs["energy"].element_references, refs.element_references
    )


def test_fit_linear_references(element_refs, dataset):
    # create the composition matrix
    energy = np.array([d.energy for d in dataset])
    cmatrix = np.vstack(
        [
            np.bincount(d.atomic_numbers.int().numpy(), minlength=MAX_NUM_ELEMENTS + 1)
            for d in dataset
        ]
    )
    mask = cmatrix.sum(axis=0) != 0.0

    # fit using numpy
    element_refs_np = np.zeros(MAX_NUM_ELEMENTS + 1)
    element_refs_np[mask] = np.linalg.lstsq(cmatrix[:, mask], energy, rcond=None)[0]

    # length is max_num_elements + 1, since H starts at 1
    assert len(element_refs["energy"].element_references) == MAX_NUM_ELEMENTS + 1
    # first element is dummy, should always be zero
    assert element_refs["energy"].element_references[0] == 0.0
    # elements not present should be zero
    npt.assert_allclose(element_refs["energy"].element_references.numpy()[~mask], 0.0)
    # torch fit vs numpy fit
    npt.assert_allclose(
        element_refs_np, element_refs["energy"].element_references.numpy(), atol=1e-5
    )
    # close enough to ground truth w/out noise
    npt.assert_allclose(
        DUMMY_EREFS[mask],
        element_refs["energy"].element_references.numpy()[mask],
        atol=5e-2,
    )
