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
    return data_list[0]


def load_model():
    torch.manual_seed(4)
    setup_imports()
    model = registry.get_model_class("escn")(
        use_pbc = True,
        use_pbc_single = False,
        regress_forces = True,
        otf_graph = True,
        max_neighbors = 20,
        cutoff = 8.0,
        max_num_elements = 90,
        num_layers = 8,
        lmax_list = [4],
        mmax_list = [2],
        sphere_channels = 128,
        hidden_channels = 256,
        edge_channels = 128,
        num_sphere_samples = 128,
        distance_function = "gaussian",
        basis_width_scalar = 1.0,
        distance_resolution = 0.02,
        show_timing_info = False,
        resolution = None,
    )
    return model

@pytest.fixture(scope="session")
def init():
    init_local_distributed_process_group()


class TestESCNCompiles:
    def escn_baseline(self, init):
        data = load_data()
        data = data_list_collater([data]).to("cuda")
        model = load_model().cuda()
        ddp_model = DistributedDataParallel(model)
        return ddp_model(data)

    def test_escn_compiles(self, init):
        data = load_data()
        data = data_list_collater([data]).to("cuda")
        model = load_model().cuda()
        ddp_model = DistributedDataParallel(model)

        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        # torch._dynamo.config.suppress_errors = True

        os.environ["TORCH_LOGS"] = "+dynamo,recompiles"
        compiled_model = torch.compile(model, dynamic=True)
        torch._dynamo.config.optimize_ddp = False
        compiled_model(data)
