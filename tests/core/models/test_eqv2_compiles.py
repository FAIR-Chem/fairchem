"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import io
import os

import pytest
import requests
import torch
from ase.io import read
from torch.nn.parallel.distributed import DistributedDataParallel

from fairchem.core.common.registry import registry
from fairchem.core.common.distutils import init_local_distributed_process_group
from fairchem.core.common.utils import load_state_dict, setup_imports
from fairchem.core.datasets import data_list_collater
from fairchem.core.preprocessing import AtomsToGraphs


def load_data():
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=":",
        format="json",
    )
    a2g = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_edges=False,
        r_fixed=True,
    )
    data_list = a2g.convert_all(atoms)
    return data_list


def load_model():
    torch.manual_seed(4)
    setup_imports()
    model = registry.get_model_class("equiformer_v2")(
        use_pbc=True,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=20,
        max_radius=12.0,
        max_num_elements=90,
        num_layers=8,
        sphere_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=64,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        norm_type="layer_norm_sh",
        lmax_list=[4],
        mmax_list=[2],
        grid_resolution=18,
        num_sphere_samples=128,
        edge_channels=128,
        use_atom_edge_embedding=True,
        distance_function="gaussian",
        num_distance_basis=512,
        attn_activation="silu",
        use_s2_act_attn=False,
        ffn_activation="silu",
        use_gate_act=False,
        use_grid_mlp=True,
        alpha_drop=0.1,
        drop_path_rate=0.1,
        proj_drop=0.0,
        weight_init="uniform",
    )
    return model


def init(backend="nccl"):
    if not torch.distributed.is_initialized():
        init_local_distributed_process_group(backend=backend)


def expected_energy_forces():
    energy = torch.tensor([-0.0261])
    forces = torch.tensor([-0.0008, -0.0018, -0.0020])
    return energy, forces


class TestEQV2Compiles:
    def eqv2_baseline_output(self, backend: str):
        init(backend=backend)
        data = load_data()
        data = data_list_collater([data[0]])#.to("cuda")
        model = load_model()#.cuda()
        ddp_model = DistributedDataParallel(model)
        return ddp_model(data)

    def test_baseline_cpu(self):
        outputs = self.eqv2_baseline_output("gloo")
        energy, forces_mean = outputs["energy"].detach().cpu(), outputs["forces"].mean(0).detach().cpu()
        expected_energy, expected_forces = expected_energy_forces()
        print(energy)
        print(forces_mean)
        assert torch.allclose(energy, expected_energy, atol=1e-4)
        assert torch.allclose(forces_mean, expected_forces, atol=1e-4)

    def test_eqv2_compiles(self):
        init()
        data = load_data()
        data0 = data_list_collater([data[0]]).to("cuda")
        data1 = data_list_collater([data[1]]).to("cuda")
        model = load_model().cuda()
        ddp_model = DistributedDataParallel(model)

        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        # torch._dynamo.config.suppress_errors = True

        os.environ["TORCH_LOGS"] = "+dynamo,recompiles"
        compiled_model = torch.compile(model, dynamic=True)
        torch._dynamo.config.optimize_ddp = False
        compiled_model(data0)
        compiled_model(data1)
        # import pdb; pdb.set_trace()
