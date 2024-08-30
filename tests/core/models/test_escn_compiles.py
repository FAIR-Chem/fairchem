"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import io
import os
import random
import numpy as np
import logging

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
from fairchem.core.common.transforms import RandomRotate

from fairchem.core.models.utils.so3_utils import CoefficientMapping
from fairchem.core.models.escn import escn_exportable

from torch.export import export
from torch.export import Dim

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="skipping when no gpu")


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


def load_model(name: str):
    torch.manual_seed(4)
    setup_imports()
    model = registry.get_model_class(name)(
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

def init(backend: str):
    if not torch.distributed.is_initialized():
        init_local_distributed_process_group(backend=backend)

class TestESCNCompiles:
    def test_escn_baseline_cpu(self, tol=1e-5):
        init('gloo')
        data = load_data()
        data = data_list_collater([data])
        base_model = DistributedDataParallel(load_model("escn"))
        export_model = DistributedDataParallel(load_model("escn_export"))
        base_output = base_model(data)
        export_output = export_model(data)
        torch.set_printoptions(precision=8)
        assert torch.allclose(base_output["energy"], export_output["energy"], atol=tol)
        assert torch.allclose(base_output["forces"].mean(0), export_output["forces"].mean(0), atol=tol)

    @skip_if_no_cuda
    def test_escn_baseline_cuda(self, tol=1e-5):
        init('nccl')
        data = load_data()
        data = data_list_collater([data]).to("cuda")
        base_model = DistributedDataParallel(load_model("escn")).cuda()
        export_model = DistributedDataParallel(load_model("escn_export")).cuda()
        base_output = base_model(data)
        export_output = export_model(data)
        torch.set_printoptions(precision=8)
        assert torch.allclose(base_output["energy"], export_output["energy"], atol=tol)
        assert torch.allclose(base_output["forces"].mean(0), export_output["forces"].mean(0), atol=tol)

    def test_escn_compiles(self):
        init("gloo")
        data = load_data()
        data = data_list_collater([data])
        model = load_model()
        ddp_model = DistributedDataParallel(model)

        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        # torch._dynamo.config.suppress_errors = True

        # os.environ["TORCH_LOGS"] = "+dynamo,recompiles"
        torch._logging.set_logs(dynamo = logging.INFO)
        # os.environ["TORCHDYNAMO_VERBOSE"] = "1"
        # os.environ["TORCHDYNAMO_REPRO_AFTER"]="dynamo"
        # torch._dynamo.config.verbose = True
        compiled_model = torch.compile(model, dynamic=True)
        torch._dynamo.config.optimize_ddp = False
        # torch._dynamo.explain(model)(data)
        # assert False
        # torch._dynamo.reset()
        # explain_output = torch._dynamo.explain(model)(data)
        # print(explain_output)

        output = compiled_model(data)
        # expected_energy, expected_forces = expected_energy_forces()
        # assert torch.allclose(output["energy"], expected_energy)
        # assert torch.allclose(output["forces"].mean(0), expected_forces)

    def test_rotation_invariance(self) -> None:
        random.seed(1)
        data = load_data()

        # Sampling a random rotation within [-180, 180] for all axes.
        transform = RandomRotate([-180, 180], [0, 1, 2])
        data_rotated, rot, inv_rot = transform(data.clone())
        assert not np.array_equal(data.pos, data_rotated.pos)

        # Pass it through the model.
        batch = data_list_collater([data, data_rotated])
        model = load_model("escn_export")
        model.eval()
        out = model(batch)

        # Compare predicted energies and forces (after inv-rotation).
        energies = out["energy"].detach()
        np.testing.assert_almost_equal(energies[0], energies[1], decimal=5)

        forces = out["forces"].detach()
        logging.info(forces)
        np.testing.assert_array_almost_equal(
            forces[: forces.shape[0] // 2],
            torch.matmul(forces[forces.shape[0] // 2 :], inv_rot),
            decimal=5,
        )

    def test_escn_so2_conv_exports(self) -> None:
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        inp1_dim0 = Dim("inp1_dim0")
        inp1_dim1 = None
        inp1_dim2 = None
        inp2_dim0 = inp1_dim0
        inp2_dim1 = None

        dynamic_shapes1 = {
            "x": {0: inp1_dim0, 1: inp1_dim1, 2: inp1_dim2},
            "x_edge": {0: inp2_dim0, 1: inp2_dim1},
        }

        lmax, mmax = 4, 2
        mappingReduced = escn_exportable.CoefficientMapping([lmax], [mmax])
        shpere_channels = 128
        edge_channels = 128
        args=(torch.rand(680, 19, shpere_channels), torch.rand(680, edge_channels))

        so2 = escn_exportable.SO2Block(
            sphere_channels=shpere_channels, 
            hidden_channels=128,
            edge_channels=edge_channels,
            lmax_list=[lmax], 
            mmax_list=[mmax], 
            act=torch.nn.SiLU(), 
            mappingReduced=mappingReduced
        )
        prog = export(so2, args=args, dynamic_shapes=dynamic_shapes1)
        export_out = prog.module()(*args)
        regular_out = so2(*args)
        assert torch.allclose(export_out, regular_out)