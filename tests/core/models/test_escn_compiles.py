"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import random
import tempfile

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
    SO3_Grid,
)
from fairchem.core.models.scn.smearing import GaussianSmearing
from fairchem.core.preprocessing import AtomsToGraphs

skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="skipping when no gpu"
)

MAX_ELEMENTS = 100
CUTOFF = 6.0
MAX_NEIGHBORS = 300


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
    return data_list_collater(data_list)


def load_model(type: str, compile=False, export=False):
    torch.manual_seed(4)
    setup_imports()
    if type == "baseline":
        return registry.get_model_class("escn")(
            use_pbc=True,
            use_pbc_single=False,
            regress_forces=True,
            max_neighbors=MAX_NEIGHBORS,
            cutoff=CUTOFF,
            max_num_elements=MAX_ELEMENTS,
            num_layers=8,
            lmax_list=[6],
            mmax_list=[0],
            sphere_channels=128,
            hidden_channels=256,
            edge_channels=128,
            num_sphere_samples=128,
            distance_function="gaussian",
            basis_width_scalar=1.0,
            distance_resolution=0.02,
            resolution=None,
        )
    elif type == "exportable":
        return registry.get_model_class("escn_export")(
            max_neighbors=MAX_NEIGHBORS,
            cutoff=CUTOFF,
            max_num_elements=MAX_ELEMENTS,
            num_layers=8,
            lmax=6,
            mmax=0,
            sphere_channels=128,
            hidden_channels=256,
            edge_channels=128,
            num_sphere_samples=128,
            distance_function="gaussian",
            basis_width_scalar=1.0,
            distance_resolution=0.02,
            resolution=None,
            compile=compile,
            export=export,
        )
    else:
        raise RuntimeError("type not recognized")


def sim_export_input(
    natoms_range: list[int] = (2, 100),
    nedges_max: int = 10000,
    batch_size: int = 2,
    atom_num_mnax: int = MAX_ELEMENTS,
    device: str = "cpu",
) -> tuple[torch.Tensor]:
    natoms_list = []
    for _ in range(batch_size):
        natoms = torch.randint(natoms_range[0], natoms_range[1], (1,)).item()
        natoms_list.append(natoms)
    natoms_total = sum(natoms_list)
    nedges = torch.randint(1, nedges_max, (1,)).item()
    pos = torch.rand(natoms_total, 3)

    batch_list = [
        torch.ones(natoms_list[i], dtype=torch.long) * i for i in range(batch_size)
    ]
    batch = torch.cat(batch_list)
    natoms = torch.tensor(natoms_list, dtype=torch.long)
    atomic_numbers = torch.randint(0, atom_num_mnax, (natoms_total,))
    edge_index = torch.randint(0, natoms_total, (2, nedges))
    distances = torch.rand(nedges)
    edge_distance_vec = torch.rand(nedges, 3)
    output = (
        pos,
        batch,
        natoms,
        atomic_numbers,
        edge_index,
        distances,
        edge_distance_vec,
    )
    return tuple(x.to(device) for x in output)


def init(backend: str):
    if not torch.distributed.is_initialized():
        init_local_distributed_process_group(backend=backend)


def check_escn_equivalent(data, model1, model2):
    output1 = model1(data)
    output2 = model2(data)
    assert torch.allclose(output1["energy"], output2["energy"])
    assert torch.allclose(output2["forces"], output2["forces"])


class TestESCNCompiles:
    def test_escn_baseline_cpu(self):
        init("gloo")
        data = load_data()
        base_model = DistributedDataParallel(load_model("baseline"))
        export_model = DistributedDataParallel(load_model("exportable"))
        check_escn_equivalent(data, base_model, export_model)

    @skip_if_no_cuda
    def test_escn_baseline_cuda(self):
        init("nccl")
        data = load_data().to("cuda")
        base_model = DistributedDataParallel(load_model("baseline")).cuda()
        export_model = DistributedDataParallel(load_model("exportable")).cuda()
        check_escn_equivalent(data, base_model, export_model)

    def test_rotation_invariance(self) -> None:
        random.seed(1)
        data = load_data()

        # Sampling a random rotation within [-180, 180] for all axes.
        transform = RandomRotate([-180, 180], [0, 1, 2])
        data_rotated, _, inv_rot = transform(data.clone())
        assert not np.array_equal(data.pos, data_rotated.pos)

        # Pass it through the model.
        batch = data_list_collater([data, data_rotated])
        model = load_model("exportable")
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
        shpere_channels = 128
        edge_channels = 128
        args = (torch.rand(680, 19, shpere_channels), torch.rand(680, edge_channels))

        so2 = escn_exportable.SO2Block(
            sphere_channels=shpere_channels,
            hidden_channels=128,
            edge_channels=edge_channels,
            lmax=lmax,
            mmax=mmax,
            act=torch.nn.SiLU(),
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
        message_block = escn_exportable.MessageBlock(
            layer_idx=0,
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            edge_channels=edge_channels,
            lmax=lmax,
            mmax=mmax,
            distance_expansion=distance_expansion,
            max_num_elements=90,
            SO3_grid=SO3_grid,
            act=torch.nn.SiLU(),
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
        layer_block = escn_exportable.LayerBlock(
            layer_idx=0,
            sphere_channels=sphere_channels,
            hidden_channels=hidden_channels,
            edge_channels=edge_channels,
            lmax=lmax,
            mmax=mmax,
            distance_expansion=distance_expansion,
            max_num_elements=90,
            SO3_grid=SO3_grid,
            act=torch.nn.SiLU(),
        )

        # generate inputs
        batch_sizes = [34, 35, 35]
        num_edges = [680, 700, 680]
        num_coefs = 25
        run_args = []
        for b, edges in zip(batch_sizes, num_edges):
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
            "wigner": {0: edges_dim, 1: None, 2: None},
        }
        exported_prog = export(
            layer_block, args=run_args[0], dynamic_shapes=dynamic_shapes1
        )
        for run_arg in run_args:
            exported_output = exported_prog.module()(*run_arg)
            compiled_model = torch.compile(layer_block, dynamic=True)
            compiled_output = compiled_model(*run_arg)
            regular_out = layer_block(*run_arg)
            assert torch.allclose(compiled_output, regular_out, atol=tol)
            assert torch.allclose(exported_output, regular_out, atol=tol)

    def test_full_escn_compiles_cpu(self):
        init("gloo")
        data = load_data()
        base_model = DistributedDataParallel(load_model("baseline"))
        exportable_model = DistributedDataParallel(
            load_model("exportable", compile=True)
        )
        check_escn_equivalent(data, exportable_model, base_model)
        # TODO: also check no recompile happens on changed inputs

    @skip_if_no_cuda
    def test_full_escn_compiles_cuda(self):
        init("nccl")
        data = load_data().to("cuda")
        base_model = DistributedDataParallel(load_model("baseline")).cuda()
        exportable_model = DistributedDataParallel(
            load_model("exportable", compile=True).cuda()
        )
        check_escn_equivalent(data, exportable_model, base_model)

    def test_full_escn_exports_static(self):
        data = sim_export_input()
        exportable_model = load_model("exportable", export=True)
        exported = export(exportable_model, args=data)
        exported.module()(*data)

    def test_full_escn_exports_with_dynamic_inputs(self):
        with tempfile.TemporaryDirectory() as tempdirname:
            torch.manual_seed(4)
            atoms_dim = Dim("atoms_dim", min=2, max=1000)
            edges_dim = Dim("edges_dim", min=2, max=10000)
            # if we want to use with variable batch dim, then we need to specify this
            # batch_dim = Dim("batch_dim")
            dynamic_shapes = {
                "pos": {0: atoms_dim, 1: None},  # second dim fixed to 3
                "batch_idx": {0: atoms_dim},
                "natoms": {0: None},
                "atomic_numbers": {0: atoms_dim},
                "edge_index": {0: None, 1: edges_dim},  # first dim fixed to 2
                "edge_distance": {0: edges_dim},
                "edge_distance_vec": {0: edges_dim, 1: None},  # second dim fixed to 3
            }

            device = "cuda" if torch.cuda.is_available() else "cpu"
            exportable_model = load_model("exportable", export=True).to(device=device)
            batch_size = 1
            example_data = sim_export_input(batch_size=batch_size, device=device)
            with torch.inference_mode():
                exported_prog = export(
                    exportable_model, args=example_data, dynamic_shapes=dynamic_shapes
                )
                # test dynamic shapes
                for _ in range(10):
                    data = sim_export_input(batch_size=batch_size, device=device)
                    exported_prog.module()(*data)

                # test saving and loading
                path = os.path.join(tempdirname, "exported_program.pt2")
                torch.export.save(exported_prog, path)
                saved_exported_program = torch.export.load(path)
                data = sim_export_input(batch_size=batch_size, device=device)
                energy_1, forces_1 = saved_exported_program.module()(*data)
                energy_2, forces_2 = exported_prog.module()(*data)
                # not exactly the same because the exported model has been ran previously, different seeds
                assert torch.allclose(energy_1, energy_2, atol=1e-5)
                assert torch.allclose(forces_1, forces_2, atol=1e-5)

                # test aot compile for C++
                so_path = os.path.join(tempdirname, "aot_export.so")
                aot_compile_options = {"aot_inductor.output_path": so_path}
                if device == "cuda":
                    aot_compile_options.update({"max_autotune": True})
                so_path = torch._export.aot_compile(
                    exportable_model,
                    args=example_data,
                    dynamic_shapes=dynamic_shapes,
                    options=aot_compile_options,
                )
                assert os.path.exists(so_path)
