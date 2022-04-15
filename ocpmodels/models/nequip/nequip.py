"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

This code borrows heavily from the Nequip implementation: https://github.com/mir-group/nequip. License:

---
MIT License

Copyright (c) 2021 The President and Fellows of Harvard College

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch

try:
    from nequip.data import AtomicData, AtomicDataDict
    from nequip.nn import (
        AtomwiseLinear,
        AtomwiseReduce,
        ConvNetLayer,
        PerSpeciesScaleShift,
        SequentialGraphNetwork,
    )
    from nequip.nn.cutoffs import PolynomialCutoff
    from nequip.nn.embedding import (
        OneHotAtomEncoding,
        RadialBasisEdgeEncoding,
        SphericalHarmonicEdgeAttrs,
    )
    from nequip.nn.radial_basis import BesselBasis
    from nequip.utils import Config, Output, instantiate
except ImportError:
    pass
from torch import nn
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)


@registry.register_model("nequip")
class NequipWrap(nn.Module):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        num_basis=8,  # number of basis functions used in radial basis
        BesselBasis_trainable=True,  # train bessel weights
        PolynomialCutoff_p=6.0,  # p-exponent used in polynomial cutoff
        # irreps used in hidden layer of output block
        conv_to_output_hidden_irreps_out="8x0e",
        # irreps for the chemical embedding of species
        chemical_embedding_irreps_out="8x0e",
        # irreps used for hidden features. Default is lmax=1, with even and odd parities
        feature_irreps_hidden="8x0o + 8x0e + 8x1o + 8x1e",
        # irreps of the spherical harmonics used for edges. If single integer, full SH up to lmax=that_integer
        irreps_edge_sh="0e + 1o",
        nonlinearity_type="gate",
        num_layers=3,  # number of interaction blocks
        invariant_layers=2,  # number of radial layers
        invariant_neurons=64,  # number of hidden neurons in radial function
        use_sc=True,  # use self-connection or not
        cutoff=4.0,
        resnet=False,
        regress_forces=False,
        direct_forces=False,
        direct_pos=False,
        otf_graph=False,
        use_pbc=False,
        max_neighbors=50,
        **convolution_args,
    ):
        super().__init__()
        self.use_pbc = use_pbc
        self.otf_graph = otf_graph
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.direct_pos = direct_pos
        self.cutoff = cutoff
        self.max_neighbors = 50
        config = {
            "BesselBasis_trainable": BesselBasis_trainable,
            "PolynomialCutoff_p": PolynomialCutoff_p,
            "conv_to_output_hidden_irreps_out": conv_to_output_hidden_irreps_out,
            "chemical_embedding_irreps_out": chemical_embedding_irreps_out,
            "feature_irreps_hidden": feature_irreps_hidden,
            "irreps_edge_sh": irreps_edge_sh,
            "nonlinearity_type": nonlinearity_type,
            "num_basis": num_basis,
            "num_layers": num_layers,
            "r_max": cutoff,
            "resnet": resnet,
            "regress_forces": regress_forces,
            "num_types": 100,
            "invariant_layers": invariant_layers,
            "invariant_neurons": invariant_neurons,
            "use_sc": use_sc,
            **convolution_args,
        }
        layers = {
            # -- Encode --
            "one_hot": OneHotAtomEncoding,
            "spharm_edges": SphericalHarmonicEdgeAttrs,
            "radial_basis": RadialBasisEdgeEncoding,
            # -- Embed features --
            "chemical_embedding": AtomwiseLinear,
        }

        # add convnet layers
        # insertion preserves order
        for layer_i in range(num_layers):
            layers[f"layer{layer_i}_convnet"] = ConvNetLayer

        layers.update(
            {
                # -- output block --
                "conv_to_output_hidden": AtomwiseLinear,
                "output_hidden_to_scalar": (
                    AtomwiseLinear,
                    dict(
                        irreps_out="1x0e",
                        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    ),
                ),
            }
        )

        layers["total_energy_sum"] = (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        )

        self.model = SequentialGraphNetwork.from_parameters(
            shared_params=config, layers=layers
        )

    @staticmethod
    def convert_ocp(data):
        data = AtomicData(
            pos=data.pos,
            edge_index=data.edge_index,
            edge_cell_shift=data.cell_offsets.float(),
            cell=data.cell,
            atom_types=data.atomic_numbers.long(),
            batch=data.batch,
            edge_vectors=data.edge_vec,
        )
        data = AtomicData.to_AtomicDataDict(data)

        return data

    def _forward(self, data):
        return self.model(data)

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)

        if self.use_pbc:
            if self.otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data, self.cutoff, self.max_neighbors
                )
            else:
                edge_index = data.edge_index
                cell_offsets = data.cell_offsets
                neighbors = data.neighbors

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=False,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_vec = out["distance_vec"]
        else:
            self.otf_graph = True
            edge_index = radius_graph(
                data.pos,
                r=self.cutoff,
                batch=data.batch,
                max_num_neighbors=self.max_neighbors,
            )
            j, i = edge_index
            edge_vec = data.pos[j] - data.pos[i]
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device
            )

        data.edge_index = edge_index
        data.cell_offsets = cell_offsets
        data.edge_vec = edge_vec

        data = self.convert_ocp(data)

        if self.direct_forces:
            forces = self.model(data)["Nx3"]

        energy = self._forward(data)["total_energy"]
        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data["pos"],
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
