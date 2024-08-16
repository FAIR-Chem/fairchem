from __future__ import annotations

from itertools import product

import pytest
import torch
from ase.build import bulk

from fairchem.core.common.utils import cg_change_mat, irreps_sum
from fairchem.core.datasets import data_list_collater
from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone
from fairchem.core.models.equiformer_v2.prediction_heads import Rank2SymmetricTensorHead
from fairchem.core.preprocessing import AtomsToGraphs


def _reshape_tensor(out, batch_size=1):
    tensor = torch.zeros((batch_size, irreps_sum(2)), requires_grad=False)
    tensor[:, max(0, irreps_sum(1)) : irreps_sum(2)] = out.view(batch_size, -1)
    tensor = torch.einsum("ba, cb->ca", cg_change_mat(2), tensor)
    return tensor.view(3, 3)


@pytest.fixture(scope="session")
def batch():
    a2g = AtomsToGraphs(r_pbc=True)
    return data_list_collater([a2g.convert(bulk("ZnFe", "wurtzite", a=2.0))])


@pytest.mark.parametrize(
    ("decompose", "edge_level_mlp", "use_source_target_embedding", "extensive"),
    list(product((True, False), repeat=4)),
)
def test_rank2_head(
    batch, decompose, edge_level_mlp, use_source_target_embedding, extensive
):
    torch.manual_seed(100)  # fix network initialization
    backbone = EquiformerV2Backbone(
        num_layers=2,
        sphere_channels=8,
        attn_hidden_channels=8,
        num_sphere_samples=8,
        edge_channels=8,
    )
    head = Rank2SymmetricTensorHead(
        backbone=backbone,
        output_name="out",
        decompose=decompose,
        edge_level_mlp=edge_level_mlp,
        use_source_target_embedding=use_source_target_embedding,
        extensive=extensive,
    )

    r2_out = head(batch, backbone(batch))

    if decompose is True:
        assert "out_isotropic" in r2_out
        assert "out_anisotropic" in r2_out
        # isotropic must be scalar
        assert r2_out["out_isotropic"].shape[1] == 1
        tensor = _reshape_tensor(r2_out["out_isotropic"])
        # anisotropic must be traceless
        assert torch.diagonal(tensor).sum().item() == pytest.approx(0.0, abs=2e-8)
    else:
        assert "out" in r2_out
        tensor = r2_out["out"].view(3, 3)

    # all tensors must be symmetric
    assert torch.allclose(tensor, tensor.transpose(0, 1))
