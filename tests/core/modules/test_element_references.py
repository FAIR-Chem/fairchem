from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import torch

from fairchem.core.datasets import data_list_collater
from fairchem.core.modules.normalization.element_references import (
    LinearReferences,
    create_element_references,
    fit_linear_references,
)


@pytest.fixture(scope="session")
def element_refs(dummy_binary_dataset, max_num_elements):
    return fit_linear_references(
        ["energy"],
        dataset=dummy_binary_dataset,
        batch_size=16,
        shuffle=False,
        max_num_elements=max_num_elements,
        seed=0,
    )


def test_apply_linear_references(
    element_refs, dummy_binary_dataset, dummy_element_refs
):
    max_noise = 0.05 * dummy_element_refs.mean()

    # check that removing element refs keeps only values within max noise
    batch = data_list_collater(list(dummy_binary_dataset), otf_graph=True)
    energy = batch.energy.clone().view(len(batch), -1)
    deref_energy = element_refs["energy"].dereference(energy, batch)
    assert all(deref_energy <= max_noise)

    # and check that we recover the total energy from applying references
    ref_energy = element_refs["energy"](deref_energy, batch)
    assert torch.allclose(ref_energy, energy)


def test_create_element_references(element_refs, tmp_path):
    # test from state dict
    sdict = element_refs["energy"].state_dict()

    refs = create_element_references(state_dict=sdict)
    assert isinstance(refs, LinearReferences)
    assert torch.allclose(
        element_refs["energy"].element_references, refs.element_references
    )

    # test from saved stated dict
    torch.save(sdict, tmp_path / "linref.pt")
    refs = create_element_references(file=tmp_path / "linref.pt")
    assert isinstance(refs, LinearReferences)
    assert torch.allclose(
        element_refs["energy"].element_references, refs.element_references
    )

    # from a legacy numpy npz file
    np.savez(
        tmp_path / "linref.npz", coeff=element_refs["energy"].element_references.numpy()
    )
    refs = create_element_references(file=tmp_path / "linref.npz")
    assert isinstance(refs, LinearReferences)
    assert torch.allclose(
        element_refs["energy"].element_references, refs.element_references
    )

    # from a numpy npz file
    np.savez(
        tmp_path / "linref.npz",
        element_references=element_refs["energy"].element_references.numpy(),
    )

    refs = create_element_references(file=tmp_path / "linref.npz")
    assert isinstance(refs, LinearReferences)
    assert torch.allclose(
        element_refs["energy"].element_references, refs.element_references
    )


def test_fit_linear_references(
    element_refs, dummy_binary_dataset, max_num_elements, dummy_element_refs
):
    # create the composition matrix
    energy = np.array([d.energy for d in dummy_binary_dataset])
    cmatrix = np.vstack(
        [
            np.bincount(d.atomic_numbers.int().numpy(), minlength=max_num_elements + 1)
            for d in dummy_binary_dataset
        ]
    )
    mask = cmatrix.sum(axis=0) != 0.0

    # fit using numpy
    element_refs_np = np.zeros(max_num_elements + 1)
    element_refs_np[mask] = np.linalg.lstsq(cmatrix[:, mask], energy, rcond=None)[0]

    # length is max_num_elements + 1, since H starts at 1
    assert len(element_refs["energy"].element_references) == max_num_elements + 1
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
        dummy_element_refs[mask],
        element_refs["energy"].element_references.numpy()[mask],
        atol=5e-2,
    )


def test_fit_seed_no_seed(dummy_binary_dataset, max_num_elements):
    refs_seed = fit_linear_references(
        ["energy"],
        dataset=dummy_binary_dataset,
        batch_size=16,
        num_batches=len(dummy_binary_dataset) // 16 - 2,
        shuffle=True,
        max_num_elements=max_num_elements,
        seed=0,
    )
    refs_seed1 = fit_linear_references(
        ["energy"],
        dataset=dummy_binary_dataset,
        batch_size=16,
        num_batches=len(dummy_binary_dataset) // 16 - 2,
        shuffle=True,
        max_num_elements=max_num_elements,
        seed=0,
    )
    refs_noseed = fit_linear_references(
        ["energy"],
        dataset=dummy_binary_dataset,
        batch_size=16,
        num_batches=len(dummy_binary_dataset) // 16 - 2,
        shuffle=True,
        max_num_elements=max_num_elements,
        seed=1,
    )

    assert torch.allclose(
        refs_seed["energy"].element_references,
        refs_seed1["energy"].element_references,
        atol=1e-6,
    )
    assert not torch.allclose(
        refs_seed["energy"].element_references,
        refs_noseed["energy"].element_references,
        atol=1e-6,
    )
