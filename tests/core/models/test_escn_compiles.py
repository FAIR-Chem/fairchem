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
from fairchem.core.models.scn.smearing import GaussianSmearing
from fairchem.core.models.base import GraphModelMixin

from fairchem.core.models.utils.so3_utils import CoefficientMapping, SO3_Grid, rotation_to_wigner
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
    def test_escn_baseline_cpu(self, tol=1e-8):
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
    def test_escn_baseline_cuda(self, tol=1e-8):
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

    def test_escn_so2_conv_exports_and_compiles(self, tol=1e-5) -> None:
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
            lmax=lmax, 
            mmax=mmax, 
            act=torch.nn.SiLU(), 
            mappingReduced=mappingReduced
        )
        prog = export(so2, args=args, dynamic_shapes=dynamic_shapes1)
        export_out = prog.module()(*args)
        regular_out = so2(*args)
        assert torch.allclose(export_out, regular_out, atol=tol)

        compiled_model = torch.compile(so2, dynamic=True)
        compiled_out = compiled_model(*args)
        assert torch.allclose(compiled_out, regular_out, atol=tol)

    def test_escn_message_block_exports_and_compiles(self, tol=1e-5) -> None:
        random.seed(1)

        sphere_channels = 128
        hidden_channels = 128
        edge_channels = 128
        lmax, mmax = 4, 2
        distance_expansion = GaussianSmearing(0.0, 8.0, int(8.0 / 0.02), 1.0)
        SO3_grid = torch.nn.ModuleDict()
        SO3_grid["lmax_lmax"] = SO3_Grid(lmax, lmax)
        SO3_grid["lmax_mmax"] = SO3_Grid(lmax, mmax)
        mappingReduced = CoefficientMapping([lmax], [mmax])
        message_block = escn_exportable.MessageBlock(
            layer_idx = 0,
            sphere_channels = sphere_channels,
            hidden_channels = hidden_channels,
            edge_channels = edge_channels,
            lmax = lmax,
            mmax = mmax,
            distance_expansion = distance_expansion,
            max_num_elements = 90,
            SO3_grid = SO3_grid,
            act = torch.nn.SiLU(),
            mappingReduced = mappingReduced
        )

        data = load_data()
        data = data_list_collater([data])
        full_model = load_model("escn_export")
        graph = full_model.generate_graph(data)
        edge_rot_mat = full_model._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )
        wigner = rotation_to_wigner(edge_rot_mat, 0, lmax).detach()

        # generate inputs
        batch_sizes = [34]
        num_coefs = 25
        num_edges = 680
        args = []
        for b in batch_sizes:
            x = torch.rand([b, num_coefs, sphere_channels])
            atom_n = torch.randint(1, 90, (b,))
            edge_d = torch.rand([num_edges])
            edge_indx = torch.randint(0, b, (2, num_edges))
            args.append((x, atom_n, edge_d, edge_indx, wigner))

        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.verbose = True
        compiled_model = torch.compile(message_block, dynamic=True)
        compiled_output = compiled_model(*args[0])

        exported_prog = export(message_block, args=args[0])
        exported_output = exported_prog(*args[0])

        regular_out = message_block(*args[0])
        assert torch.allclose(compiled_output, regular_out, atol=tol)
        assert torch.allclose(exported_output, regular_out, atol=tol)

    def test_escn_layer_block_exports_and_compiles(self, tol=1e-5) -> None:
        random.seed(1)

        sphere_channels = 128
        hidden_channels = 128
        edge_channels = 128
        lmax, mmax = 4, 2
        distance_expansion = GaussianSmearing(0.0, 8.0, int(8.0 / 0.02), 1.0)
        SO3_grid = torch.nn.ModuleDict()
        SO3_grid["lmax_lmax"] = SO3_Grid(lmax, lmax)
        SO3_grid["lmax_mmax"] = SO3_Grid(lmax, mmax)
        mappingReduced = CoefficientMapping([lmax], [mmax])
        layer_block = escn_exportable.LayerBlock(
            layer_idx = 0,
            sphere_channels = sphere_channels,
            hidden_channels = hidden_channels,
            edge_channels = edge_channels,
            lmax = lmax,
            mmax = mmax,
            distance_expansion = distance_expansion,
            max_num_elements = 90,
            SO3_grid = SO3_grid,
            act = torch.nn.SiLU(),
            mappingReduced = mappingReduced
        )

        # generate inputs
        batch_sizes = [34, 35, 35]
        num_edges = [680, 700, 680]
        num_coefs = 25
        run_args = []
        for b,edges in zip(batch_sizes, num_edges):
            x = torch.rand([b, num_coefs, sphere_channels])
            atom_n = torch.randint(1, 90, (b,))
            edge_d = torch.rand([edges])
            edge_indx = torch.randint(0, b, (2, edges))
            wigner = torch.rand([edges, num_coefs, num_coefs])
            run_args.append((x, atom_n, edge_d, edge_indx, wigner))

        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.verbose = True
        # torch._logging.set_logs(dynamo = logging.INFO)
        # torch._dynamo.reset()
        # explain_output = torch._dynamo.explain(message_block)(*args[0])
        # print(explain_output)

        batch_dim = Dim("batch_dim")
        edges_dim = Dim("edges_dim")
        dynamic_shapes1 = {
            "x": {0: batch_dim, 1: None, 2: None},
            "atomic_numbers": {0: batch_dim},
            "edge_distance": {0: edges_dim},
            "edge_index": {0: None, 1: edges_dim},
            "wigner": {0: edges_dim, 1: None, 2: None} 
        }
        exported_prog = export(layer_block, args=run_args[0], dynamic_shapes=dynamic_shapes1)
        for run_arg in run_args:
            exported_output = exported_prog(*run_arg)
            compiled_model = torch.compile(layer_block, dynamic=True)
            compiled_output = compiled_model(*run_arg)
            regular_out = layer_block(*run_arg)
            assert torch.allclose(compiled_output, regular_out, atol=tol)
            assert torch.allclose(exported_output, regular_out, atol=tol)

    def test_escn_compiles(self):
        init("gloo")
        data = load_data()
        data = data_list_collater([data])
        model = load_model('escn_export')
        ddp_model = DistributedDataParallel(model)

        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        # torch._dynamo.config.suppress_errors = True

        # os.environ["TORCH_LOGS"] = "+dynamo,recompiles"
        # torch._logging.set_logs(dynamo = logging.INFO)
        # os.environ["TORCHDYNAMO_VERBOSE"] = "1"
        # os.environ["TORCHDYNAMO_REPRO_AFTER"]="dynamo"
        # torch._dynamo.config.verbose = True
        compiled_model = torch.compile(model, dynamic=True)
        # torch._dynamo.explain(model)(data)
        # assert False
        # torch._dynamo.reset()
        # explain_output = torch._dynamo.explain(model)(data)
        # print(explain_output)

        output = compiled_model(data)
        # expected_energy, expected_forces = expected_energy_forces()
        # assert torch.allclose(output["energy"], expected_energy)
        # assert torch.allclose(output["forces"].mean(0), expected_forces)
