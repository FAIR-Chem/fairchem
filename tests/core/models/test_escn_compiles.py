"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import pytest
import torch
from ase.io import read
from torch.export import Dim, export
from torch.nn.parallel.distributed import DistributedDataParallel

from fairchem.core.common.registry import registry
from fairchem.core.common.test_utils import init_local_distributed_process_group
from fairchem.core.common.transforms import RandomRotate
from fairchem.core.common.utils import setup_imports
from fairchem.core.datasets import data_list_collater
from fairchem.core.models.escn import escn_exportable
from fairchem.core.models.escn.so3_exportable import (
    CoefficientMapping,
    SO3_Grid,
)
from fairchem.core.models.scn.smearing import GaussianSmearing
from fairchem.core.preprocessing import AtomsToGraphs

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="skipping when no gpu")


def load_data():
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    a2g = AtomsToGraphs(
        max_neigh=300,
        radius=6.0,
        r_edges=True,
        r_fixed=True,
        r_distances=True,
    )
    data_list = a2g.convert_all([atoms])
    return data_list[0]


def load_escn_model():
    torch.manual_seed(4)
    setup_imports()
    return registry.get_model_class("escn")(
        use_pbc = True,
        use_pbc_single = False,
        regress_forces = True,
        max_neighbors = 300,
        cutoff = 6.0,
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
        resolution = None,
    )

def load_escn_exportable_model():
    torch.manual_seed(4)
    setup_imports()
    return registry.get_model_class("escn_export")(
        regress_forces = True,
        cutoff = 6.0,
        max_num_elements = 90,
        num_layers = 8,
        lmax = 4,
        mmax = 2,
        sphere_channels = 128,
        hidden_channels = 256,
        edge_channels = 128,
        num_sphere_samples = 128,
        distance_function = "gaussian",
        basis_width_scalar = 1.0,
        distance_resolution = 0.02,
        resolution = None,
    )

def init(backend: str):
    if not torch.distributed.is_initialized():
        init_local_distributed_process_group(backend=backend)

class TestESCNCompiles:
    def test_escn_baseline_cpu(self, tol=1e-8):
        init("gloo")
        data = load_data()
        data_tg = data_list_collater([data])
        data_export = data_list_collater([data], to_dict=True)

        base_model = DistributedDataParallel(load_escn_model())
        export_model = DistributedDataParallel(load_escn_exportable_model())
        base_output = base_model(data_tg)
        export_output = export_model(data_export)
        torch.set_printoptions(precision=8)
        assert torch.allclose(base_output["energy"], export_output["energy"], atol=tol)
        assert torch.allclose(base_output["forces"].mean(0), export_output["forces"].mean(0), atol=tol)

    @skip_if_no_cuda
    def test_escn_baseline_cuda(self, tol=1e-8):
        init("nccl")
        data = load_data()
        data_tg = data_list_collater([data]).to("cuda")
        data_export = data_list_collater([data], to_dict=True)
        data_export_cu = {k:v.to("cuda") for k,v in data_export.items()}

        base_model = DistributedDataParallel(load_escn_model().cuda())
        export_model = DistributedDataParallel(load_escn_exportable_model().cuda())
        base_output = base_model(data_tg)
        export_output = export_model(data_export_cu)
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
        batch = data_list_collater([data, data_rotated], to_dict=True)
        model = load_escn_exportable_model()
        model.eval()
        out = model(batch)

        # Compare predicted energies and forces (after inv-rotation).
        energies = out["energy"].detach()
        np.testing.assert_almost_equal(energies[0], energies[1], decimal=7)

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

        # generate inputs
        batch_sizes = [34]
        num_coefs = 25
        num_edges = 2000
        wigner = torch.rand([num_edges, num_coefs, num_coefs])
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
        exported_output = exported_prog.module()(*args[0])

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
            exported_output = exported_prog.module()(*run_arg)
            compiled_model = torch.compile(layer_block, dynamic=True)
            compiled_output = compiled_model(*run_arg)
            regular_out = layer_block(*run_arg)
            assert torch.allclose(compiled_output, regular_out, atol=tol)
            assert torch.allclose(exported_output, regular_out, atol=tol)

    def test_full_escn_compiles(self, tol=1e-5):
        init("gloo")
        data = load_data()
        regular_data = data_list_collater([data])
        compile_data = data_list_collater([data], to_dict=True)
        escn_model = DistributedDataParallel(load_escn_model())
        exportable_model = load_escn_exportable_model()

        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        compiled_model = torch.compile(exportable_model, dynamic=True)
        output = compiled_model(compile_data)
        expected_output = escn_model(regular_data)
        assert torch.allclose(expected_output["energy"], output["energy"], atol=tol)
        assert torch.allclose(expected_output["forces"].mean(0), output["forces"].mean(0), atol=tol)

    def test_full_escn_exports(self):
        init("gloo")
        data = load_data()
        regular_data = data_list_collater([data])
        export_data = data_list_collater([data], to_dict=True)
        escn_model = load_escn_model()
        exportable_model = load_escn_exportable_model()

        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        # torch._logging.set_logs(dynamo = logging.INFO)
        # torch._dynamo.reset()
        # explained_output = torch._dynamo.explain(model)(*data)
        # print(explained_output)
        # TODO: add dynamic shapes
        exported_prog = export(exportable_model, args=(export_data,))
        export_output = exported_prog.module()(export_data)
        expected_output = escn_model(regular_data)
        assert torch.allclose(export_output["energy"], expected_output["energy"])
        assert torch.allclose(export_output["forces"].mean(0), expected_output["forces"].mean(0))
