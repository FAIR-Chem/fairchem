from __future__ import annotations

import pytest
import torch
from torch import nn

from fairchem.core.modules.loss import (
    DDPLoss,
    MAELoss,
    MSELoss,
    P2NormLoss,
    PerAtomMAELoss,
)


@pytest.fixture()
def energy():
    # batch size = 4
    pred = torch.rand([4, 1])
    target = torch.rand([4, 1])
    return pred, target


@pytest.fixture()
def forces():
    # batch size = 4
    # total atoms = 100
    pred = torch.rand(100, 3)
    target = torch.rand(100, 3)
    return pred, target


@pytest.fixture()
def natoms():
    # batch size = 4
    return torch.tensor([25, 34, 21, 20])


def test_mae(energy, forces, natoms):
    loss = MAELoss()
    ref_loss = nn.L1Loss(reduction="none")
    pred, target = energy
    assert torch.equal(loss(pred, target, natoms), ref_loss(pred, target))
    pred, target = forces
    assert torch.equal(loss(pred, target, natoms), ref_loss(pred, target))


def test_mse(energy, forces):
    loss = MSELoss()
    ref_loss = nn.MSELoss(reduction="none")
    pred, target = energy
    assert torch.equal(loss(pred, target, natoms), ref_loss(pred, target))
    pred, target = forces
    assert torch.equal(loss(pred, target, natoms), ref_loss(pred, target))


def test_per_atom_mae(energy, natoms):
    loss = PerAtomMAELoss()
    ref_loss = nn.L1Loss(reduction="none")
    pred, target = energy
    _natoms = torch.reshape(natoms, target.shape)
    assert target.shape == (target / _natoms).shape
    assert torch.equal(
        loss(pred, target, natoms), ref_loss(pred / _natoms, target / _natoms)
    )


def test_p2norm(forces, natoms):
    loss = P2NormLoss()
    pred, target = forces
    ref_norm = torch.linalg.vector_norm(pred - target, ord=2, dim=-1)
    assert torch.equal(loss(pred, target, natoms), ref_norm)


def test_mae_ddp_reduction(energy, natoms):
    # this is testing on a single process i.e. world_size=1
    # mean reduction
    loss = DDPLoss(loss_name="mae", reduction="mean")
    ref_loss = nn.L1Loss(reduction="mean")
    pred, target = energy
    assert torch.equal(loss(pred, target, natoms), ref_loss(pred, target))
    # sum reduction
    loss = DDPLoss(loss_name="mae", reduction="sum")
    ref_loss = nn.L1Loss(reduction="sum")
    assert torch.equal(loss(pred, target, natoms), ref_loss(pred, target))


def test_p2norm_ddp_reduction(forces, natoms):
    # this is testing on a single process i.e. world_size=1
    # mean reduction
    loss = DDPLoss(loss_name="p2norm", reduction="mean")
    pred, target = forces
    ref_norm = torch.linalg.vector_norm(pred - target, ord=2, dim=-1)
    ref_loss = ref_norm.mean()
    assert torch.equal(loss(pred, target, natoms), ref_loss)
    # sum reduction
    loss = DDPLoss(loss_name="p2norm", reduction="sum")
    ref_loss = ref_norm.sum()
    assert torch.equal(loss(pred, target, natoms), ref_loss)


def test_mse_ddp_reduction(forces, natoms):
    # this is testing on a single process i.e. world_size=1
    # mean reduction
    loss = DDPLoss(loss_name="mse", reduction="mean")
    ref_loss = nn.MSELoss(reduction="mean")
    pred, target = forces
    assert torch.equal(loss(pred, target, natoms), ref_loss(pred, target))
    # sum reduction
    loss = DDPLoss(loss_name="mse", reduction="sum")
    ref_loss = nn.MSELoss(reduction="sum")
    assert torch.equal(loss(pred, target, natoms), ref_loss(pred, target))
