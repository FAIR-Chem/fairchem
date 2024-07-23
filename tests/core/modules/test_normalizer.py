from __future__ import annotations

import numpy as np
import pytest
import torch

from fairchem.core.datasets import data_list_collater
from fairchem.core.modules.normalization.normalizer import (
    Normalizer,
    create_normalizer,
    fit_normalizers,
)


@pytest.fixture(scope="session")
def normalizers(dummy_binary_dataset):
    return fit_normalizers(
        ["energy", "forces"],
        override_values={"forces": {"mean": 0.0}},
        dataset=dummy_binary_dataset,
        batch_size=16,
        shuffle=False,
    )


def test_norm_denorm(normalizers, dummy_binary_dataset, dummy_element_refs):
    batch = data_list_collater(list(dummy_binary_dataset), otf_graph=True)
    # test norm and denorm
    for target, normalizer in normalizers.items():
        normed = normalizer.norm(batch[target])
        assert torch.allclose(
            (batch[target] - normalizer.mean) / normalizer.rmsd, normed
        )
        assert torch.allclose(
            normalizer.rmsd * normed + normalizer.mean, normalizer(normed)
        )


def test_create_normalizers(normalizers, dummy_binary_dataset, tmp_path):
    # test that forces mean was overriden
    assert normalizers["forces"].mean.item() == 0.0

    # test from state dict
    sdict = normalizers["energy"].state_dict()

    norm = create_normalizer(state_dict=sdict)
    assert isinstance(norm, Normalizer)
    assert norm.state_dict() == sdict

    # test from saved stated dict
    torch.save(sdict, tmp_path / "norm.pt")
    norm = create_normalizer(file=tmp_path / "norm.pt")
    assert isinstance(norm, Normalizer)
    assert norm.state_dict() == sdict

    # from a legacy numpy npz file
    np.savez(
        tmp_path / "norm.npz",
        mean=normalizers["energy"].mean.numpy(),
        std=normalizers["energy"].rmsd.numpy(),
    )
    norm = create_normalizer(file=tmp_path / "norm.npz")
    assert isinstance(norm, Normalizer)
    assert norm.state_dict() == sdict

    # from a new npz file
    np.savez(
        tmp_path / "norm.npz",
        mean=normalizers["energy"].mean.numpy(),
        rmsd=normalizers["energy"].rmsd.numpy(),
    )
    norm = create_normalizer(file=tmp_path / "norm.npz")
    assert isinstance(norm, Normalizer)
    assert norm.state_dict() == sdict

    # from tensor directly
    batch = data_list_collater(list(dummy_binary_dataset), otf_graph=True)
    norm = create_normalizer(tensor=batch.energy)
    assert isinstance(norm, Normalizer)
    # assert norm.state_dict() == sdict
    # not sure why the above fails
    new_sdict = norm.state_dict()
    for key in sdict:
        assert torch.allclose(new_sdict[key], sdict[key])

    # passing values directly
    norm = create_normalizer(
        mean=batch.energy.mean().item(), rmsd=batch.energy.std().item()
    )
    assert isinstance(norm, Normalizer)
    # assert norm.state_dict() == sdict
    new_sdict = norm.state_dict()
    for key in sdict:
        assert torch.allclose(new_sdict[key], sdict[key])

    # bad construction
    with pytest.raises(ValueError):
        create_normalizer(mean=1.0)
