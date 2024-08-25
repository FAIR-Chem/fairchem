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
from torch_geometric.data import Data, Batch

from fairchem.core.models.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding
from fairchem.core.models.equiformer_v2.so2_ops import SO2_Convolution, SO2_Convolution_Exportable

from torch.export import export
from torch.export import Dim

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


def rand_input(natoms: int) -> BaseData:
    data = Data(natoms=natoms, 
                pos=torch.rand(natoms, 3),
                cell=torch.rand([1, 3, 3]),
                atomic_numbers=torch.randint(1, 99, (1, 3, 3))
                )
    batch = Batch.from_data_list([data])
    return batch

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
        torch._dynamo.config.cache_size_limit = 1
        # torch._dynamo.config.suppress_errors = True

        os.environ["TORCH_LOGS"] = "+dynamo,recompiles"
        compiled_model = torch.compile(model, dynamic=True)
        torch._dynamo.config.optimize_ddp = False
        compiled_model(data0)
        compiled_model(data1)
        # import pdb; pdb.set_trace()

class TestExportableEQV2:
    def test_so2_conv_equivalent(self):
        torch.manual_seed(4)
        lmax, mmax = 4, 2
        sc, mc = 128, 128
        mappingReduced = CoefficientMappingModule([lmax], [mmax])

        start_rng_state = torch.random.get_rng_state()
        so2_export = SO2_Convolution_Exportable(sphere_channels=sc, m_output_channels=mc, lmax_list=[lmax], mmax_list=[mmax],mappingReduced=mappingReduced)
        torch.random.set_rng_state(start_rng_state)
        so2 = SO2_Convolution(sphere_channels=sc, m_output_channels=mc, lmax_list=[lmax], mmax_list=[mmax],mappingReduced=mappingReduced)

        inputs_tensor = (torch.rand(129, 19, 128), torch.rand(129, 856))
        inputs_embedding = SO3_Embedding(129, [lmax], 128, inputs_tensor[0].device, inputs_tensor[0].dtype)
        inputs_embedding.set_embedding(inputs_tensor[0])
        assert torch.allclose(inputs_tensor[0], inputs_embedding.embedding) 
        output = so2(inputs_embedding, inputs_tensor[1])
        output_export = so2_export(*inputs_tensor)
        assert torch.allclose(output.embedding, output_export)

    def test_so2_conv_exportable(self):
        torch._dynamo.config.assume_static_by_default = False
        torch._dynamo.config.automatic_dynamic_shapes = True
        inp1_dim0 = Dim("inp1_dim0")
        inp1_dim1 = None
        inp1_dim2 = None
        inp2_dim0 = inp1_dim0
        inp2_dim1 = Dim("inp2_dim1")

        dynamic_shapes1 = {
            "x_emb": {0: inp1_dim0, 1: inp1_dim1, 2: inp1_dim2},
            "x_edge": {0: inp2_dim0, 1: inp2_dim1},
        }

        lmax, mmax = 4, 2
        mappingReduced = CoefficientMappingModule([lmax], [mmax])
        args=(torch.rand(129, 19, 128), torch.rand(129, 856))

        so2 = SO2_Convolution_Exportable(sphere_channels=128, m_output_channels=128, lmax_list=[lmax], mmax_list=[mmax],mappingReduced=mappingReduced)
        prog = export(so2, args=args, dynamic_shapes=dynamic_shapes1)
        export_out = prog.module()(*args)
        regular_out = so2(*args)
        assert torch.allclose(export_out, regular_out)

        args2=(torch.rand(130, 19, 128), torch.rand(130, 856))
        export_out2 = prog.module()(*args2)
        regular_out2 = so2(*args2)
        assert torch.allclose(export_out2, regular_out2)

        
