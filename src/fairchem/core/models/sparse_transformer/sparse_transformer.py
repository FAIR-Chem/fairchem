"""
Author: Ryan Liu
"""
import logging
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel

from scipy.spatial.transform import Rotation as R

from .attn import SelfAttentionLayer
from .output import OutputModule
from .pair_embed import PairEmbed, EfficientPairEmbed

logger = logging.getLogger(__name__)

@registry.register_model("sparse_transformer")
class SparseTransformer(BaseModel):
    """
    Sparse Transformer

    Parameters
    ----------
    elements: List[int]
        list of possible atomic numbers
    embed_dim: int
        the dimensionality of embedding used
    hidden_dim: int
        the hidde channels of feed forward
    dropout: float
        dropout rate
    num_layers: int
        the number of layers to use
    num_heads: int
        number of attention heads
    otf_graph: bool
        calculate graph on-the-fly
    rbf_radius: float
        rbf radius
    use_pbc: bool
        to use periodic boundary condition or not
    pair_embed_style: str
        can be either "default" or "efficient"
    num_pair_embed_layers: int
        number of pair embedding layers to use
    use_vec_hat: bool
        use vec_hat as part of the input to value projection
    max_neighbors: int
        maximum number of neighbors
    num_gaussians: int
        number of gaussians to use in rbf
    trainable_rbf: bool
        use trainable RBFs
    output_layers: int
        number of output layers to use
    avg_atoms: float
        average number of atoms
    """
    def __init__(
            self,
            num_atoms: int,  # not used
            bond_feat_dim: int,  # not used
            num_targets: int,  # not used
            elements: Union[int, List[int]] = 100,
            embed_dim: int = 128,
            hidden_dim: int = 128,
            dropout: float = 0.,
            num_layers: int = 12,
            num_heads: int = 8,
            otf_graph: bool = True,
            rbf_radius: float = 5.0,
            use_pbc: bool = False,
            num_gaussians: int = 50,
            pair_embed_style: str = "default",
            num_pair_embed_layers: int = 2,
            use_vec_hat: bool = False,
            output_layers: int = 3,
            avg_atoms: float = 60,
        ):

        super().__init__()

        self.otf_graph = otf_graph
        self.cutoff = rbf_radius
        self.use_pbc = use_pbc
        self.num_layers = num_layers
        self.pair_embed_style = pair_embed_style
        self.use_vec_hat = use_vec_hat

        assert pair_embed_style in ["default", "efficient"]

        if isinstance(elements, int):
            self.register_buffer("atomic_number_mask", torch.arange(elements + 1))
        elif isinstance(elements, list):
            atomic_number_mask = - torch.ones(max(elements) + 1, dtype=torch.long)
            atomic_number_mask[elements] = torch.arange(len(elements))
            self.register_buffer("atomic_number_mask", atomic_number_mask)
        else:
            raise TypeError("elements must be integer or list of integers")

        self.atomic_number_encoder = nn.Embedding(
            len(elements),
            embed_dim,
        )
        if pair_embed_style == "default":
            self.pair_embed = PairEmbed(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_masks=num_layers,
                num_gaussians=num_gaussians,
                rbf_radius=rbf_radius,
                dropout=dropout,
                num_layers=num_pair_embed_layers,
            )
        elif pair_embed_style == "efficient":
            self.pair_embed = EfficientPairEmbed(
                num_elements=len(elements),
                num_heads=num_heads,
                num_masks=num_layers,
                num_gaussians=num_gaussians,
                rbf_radius=rbf_radius,
            )

        self.layers = nn.ModuleList([
            SelfAttentionLayer(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_vec_hat=use_vec_hat
            ) for _ in range(num_layers)
        ])

        self.init_output = OutputModule(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            hidden_layers=output_layers,
            num_gaussians=num_gaussians,
            rbf_radius=rbf_radius,
            avg_len=avg_atoms,
        )

        self.outputs = nn.ModuleList([
            OutputModule(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                hidden_layers=output_layers,
                num_gaussians=num_gaussians,
                rbf_radius=rbf_radius,
                avg_len=avg_atoms,
            ) for _ in range(num_layers)
        ])        

    def forward(self, data):

        # extract data
        batch = data.batch
        atomic_numbers = self.atomic_number_mask[data.atomic_numbers.long()]

        # build graph on-the-fly
        (
            edge_index,
            dist,
            vec,
            *_ #unused
        ) = self.generate_graph(
            data, 
            max_neighbors=data.natoms.max(),
            enforce_max_neighbors_strictly=False
        )

        # add loops
        edge_index = torch.cat([
            edge_index, 
            torch.arange(batch.size(0), device=edge_index.device).expand(2, -1)
        ], dim = 1)
        dist = torch.cat([dist, torch.zeros(batch.size(0), device=dist.device)], dim=0)
        vec = torch.cat([vec, torch.zeros((batch.size(0), 3), device=vec.device)], dim=0)

        # sort them to give better caching performance
        indx = torch.argsort(edge_index[0])
        edge_index, dist, vec = edge_index[:, indx], dist[indx], vec[indx]
        vec_hat = F.normalize(vec, dim=-1)

        # randomly rotate vec_hat for data augmentation
        if self.training:
            rotations = torch.tensor(
                R.random(batch.max() + 1).as_matrix(),
                dtype=torch.float,
                device=vec_hat.device
            )
            rot_vec_hat = torch.bmm(rotations[batch[edge_index[0]]], vec_hat[..., None]).squeeze(-1)
        else:
            rot_vec_hat = vec_hat

        # initialize inputs
        x = self.atomic_number_encoder(atomic_numbers)

        # initialize output
        energy, forces = self.init_output(x, edge_index, batch, dist, vec_hat)

        # get pair embeddings
        if self.pair_embed_style == "default":
            attn_masks = self.pair_embed(x, edge_index, dist)
        elif self.pair_embed_style == "efficient":
            attn_masks = self.pair_embed(atomic_numbers, edge_index, dist)

        # forward passing
        for i in range(self.num_layers):
            # attention block
            x = self.layers[i](x, edge_index, attn_masks[i], rot_vec_hat)

            # residual outputs
            e, f = self.outputs[i](x, edge_index, batch, dist, vec_hat)
            energy += e
            forces += f

        energy = energy / (self.num_layers + 1)
        forces = forces / (self.num_layers + 1)

        return {"energy": energy, "forces": forces, "regularizer": None}
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @torch.jit.ignore
    def no_weight_decay(self):
        # no weight decay on layer norms and embeddings
        # ref: https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm, EfficientPairEmbed)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)
