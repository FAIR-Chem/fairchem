"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml
from ase.io import read

from fairchem.core.common.registry import registry
from fairchem.core.datasets import data_list_collater
from fairchem.core.models.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
)
from fairchem.core.preprocessing import AtomsToGraphs


class TestMPrimaryLPrimary:
    def test_mprimary_lprimary_mappings(self):
        def sign(x):
            return 1 if x >= 0 else -1

        device = torch.device("cpu")
        lmax_list = [6, 8]
        mmax_list = [3, 6]
        for lmax in lmax_list:
            for mmax in mmax_list:
                c = CoefficientMappingModule([lmax], [mmax])

                embedding = SO3_Embedding(
                    length=1,
                    lmax_list=[lmax],
                    num_channels=1,
                    device=device,
                    dtype=torch.float32,
                )

                """
                Generate L_primary matrix
                L0: 0.00 ~ L0M0
                L1: -1.01 1.00 1.01 ~ L1M(-1),L1M0,L1M1
                L2: -2.02 -2.01 2.00 2.01 2.02 ~ L2M(-2),L2M(-1),L2M0,L2M1,L2M2
                """
                test_matrix_lp = []
                for l in range(lmax + 1):
                    max_m = min(l, mmax)
                    for m in range(-max_m, max_m + 1):
                        v = l * sign(m) + 0.01 * m  # +/- l . 00 m
                        test_matrix_lp.append(v)

                test_matrix_lp = (
                    torch.tensor(test_matrix_lp).reshape(1, -1, 1).to(torch.float32)
                )

                """
                Generate M_primary matrix
                M0: 0.00 , 1.00, 2.00, ... , LMax ~ M0L0, M0L1, .., M0L(LMax)
                M1: 1.01, 2.01, .., LMax.01, -1.01, -2.01, -LMax.01 ~ L1M1, L2M1, .., L(LMax)M1, L1M(-1), L2M(-1), ... , L(LMax)M(-1)
                """
                test_matrix_mp = []
                for m in range(max_m + 1):
                    for l in range(m, lmax + 1):
                        v = l + 0.01 * m  # +/- l . 00 m
                        test_matrix_mp.append(v)
                    if m > 0:
                        for l in range(m, lmax + 1):
                            v = -(l + 0.01 * m)  # +/- l . 00 m
                            test_matrix_mp.append(v)

                test_matrix_mp = (
                    torch.tensor(test_matrix_mp).reshape(1, -1, 1).to(torch.float32)
                )

                embedding.embedding = test_matrix_lp.clone()

                embedding._m_primary(c)
                mp = embedding.embedding.clone()
                (test_matrix_mp == mp).all()

                embedding._l_primary(c)
                lp = embedding.embedding.clone()
                (test_matrix_lp == lp).all()


def _load_hydra_model():
    torch.manual_seed(4)
    with open(
        Path("tests/core/models/test_configs/test_equiformerv2_hydra.yml")
    ) as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
        model = registry.get_model_class("hydra")(
            yaml_config["model"]["backbone"], yaml_config["model"]["heads"]
        )
    model.backbone.num_layers = 1
    return model


def test_eqv2_hydra_activation_checkpoint():
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    a2g = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_edges=False,
        r_fixed=True,
    )
    data_list = a2g.convert_all([atoms])
    inputs = data_list_collater(data_list)
    no_ac_model = _load_hydra_model()
    ac_model = _load_hydra_model()
    ac_model.backbone.activation_checkpoint = True

    # to do this test we need both models to have the exact same state and the only
    # way to do this is save the rng state and reset it after stepping the first model
    start_rng_state = torch.random.get_rng_state()
    outputs_no_ac = no_ac_model(inputs)
    torch.autograd.backward(
        outputs_no_ac["energy"]["energy"].sum()
        + outputs_no_ac["forces"]["forces"].sum()
    )

    # reset the rng state to the beginning
    torch.random.set_rng_state(start_rng_state)
    outptuts_ac = ac_model(inputs)
    torch.autograd.backward(
        outptuts_ac["energy"]["energy"].sum() + outptuts_ac["forces"]["forces"].sum()
    )

    # assert all the gradients are identical between the model with checkpointing and no checkpointing
    ac_model_grad_dict = {
        name: p.grad for name, p in ac_model.named_parameters() if p.grad is not None
    }
    no_ac_model_grad_dict = {
        name: p.grad for name, p in no_ac_model.named_parameters() if p.grad is not None
    }
    for name in no_ac_model_grad_dict:
        assert torch.allclose(
            no_ac_model_grad_dict[name], ac_model_grad_dict[name], atol=1e-4
        )
