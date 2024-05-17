"""
Author: Ryan Liu
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairchem.core.common.registry import registry
from fairchem.core.models.base import BaseModel

from torch_scatter import scatter

logger = logging.getLogger(__name__)

@registry.register_model("transformer")
class Transformer(BaseModel):
    """
    Pure Transformer

    Parameters
    ----------
    num_elements: int
        the number of possible elements
    emb_dim: int
        the dimensionality of embedding used
    ff_dim: int
        the hidde channels of feed forward
    num_layers: int
        the number of layers to use
    dropout: float
        dropout rate
    heads: int
        number of attention heads
    multipole: bool
        to use multipole expansion
    otf_graph: bool
        calculate graph on-the-fly
    rbf_radius: float
        rbf radius
    num_gaussians: int
        number of gaussians to use in rbf
    """
    def __init__(
            self,
            num_atoms: int,  # not used
            bond_feat_dim: int,  # not used
            num_targets: int,  # not used
            num_elements: int = 100,
            emb_dim: int = 512,
            ff_dim: int = 1024,
            num_layers: int = 12,
            dropout: float = 0.,
            heads: int = 8,
            multipole: bool = False,
            otf_graph: bool = False,
            rbf_radius: float = 5.0,
            num_gaussians: int = 50,
        ):

        super().__init__()

        self.multipole = multipole
        self.otf_graph=True
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_dim)
        )

        self.atomic_number_encoder = nn.Embedding(
            num_elements,
            emb_dim,
            padding_idx=0
        )

        self.pair_embedding = PairEmbed(
            ff_dim=ff_dim,
            dropout=dropout,
            num_elements=num_elements,
            num_head=heads,
            num_layers=num_layers,
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians
        )

        self.layers = nn.ModuleList([
            SelfAttentionLayer(
                emb_dim,
                ff_dim,
                heads,
                dropout,
            ) for _ in range(num_layers)
        ])

        self.output_modules = nn.ModuleList([
            OutputModule(
                emb_dim=emb_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                multipole=multipole
            ) for _ in range(num_layers + 1)
        ])

    def forward(self, data):

        # extract data
        pos = data.pos
        batch = data.batch
        natoms = data.natoms
        atomic_numbers = data.atomic_numbers.long()

        # tokenize the sequence
        nmax = natoms.max()
        batch_size = len(natoms)
        token_pos = torch.arange(nmax, device=natoms.device)[None, :].repeat(batch_size, 1)
        padded_node_mask = token_pos < natoms[:, None]

        padded_pos = torch.zeros((batch_size, nmax, 3), device=pos.device, dtype=torch.float)
        padded_pos[padded_node_mask] = pos
        padded_pos = padded_pos.transpose(0, 1).contiguous()

        padded_anum = torch.zeros((batch_size, nmax), device=natoms.device, dtype=torch.long)
        padded_anum[padded_node_mask] = atomic_numbers
        padded_anum = padded_anum.transpose(0, 1).contiguous()

        # encode the sequence
        x = self.pos_encoder(padded_pos) + self.atomic_number_encoder(padded_anum)
        key_padding_mask = ~padded_node_mask

        # pair embed
        attn_masks, pairs = self.pair_embedding(padded_pos, padded_anum, padded_node_mask)

        # initialize outputs
        energy, forces = self.output_modules[0](
            x,
            pairs,
            padded_pos,
            padded_node_mask,
        )

        # transformer layers
        for layer, output_layer, attn_mask in zip(self.layers, self.output_modules[1:], attn_masks):
            x = layer(x, key_padding_mask, attn_mask)
            E, F = output_layer(
                x,
                pairs,
                padded_pos,
                padded_node_mask,
            )
            energy += E
            forces += F

        return {"energy": energy, "forces": forces}
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
class OutputModule(nn.Module):
    def __init__(
        self,
        emb_dim: int = 512,
        ff_dim: int = 1024,
        dropout: float = 0,
        multipole: bool = False
    ):
        super().__init__()

        self.src_encoder = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim)
        )

        self.dst_encoder = nn.Sequential(
            nn.Linear(emb_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim)
        )

        self.energy_out = nn.Sequential(
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1)
        )
        self.energy_scale = nn.Parameter(torch.zeros(1))

        if multipole:
            self.forces_out = nn.Sequential(
                nn.Linear(ff_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, 4) # for now, use dipole for proof-of-concept
            )
        else:
            self.forces_out = nn.Sequential(
                nn.Linear(ff_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, 1)
            )
        self.forces_scale = nn.Parameter(torch.zeros(1))
        
        self.multipole = multipole

    def forward(
            self, 
            x,
            pairs,
            padded_pos,
            padded_node_mask,
        ):

        # prepare inputs
        entries = padded_node_mask.T[:, None] & padded_node_mask.T[None, :] # [L, L, N]
        entries = entries[..., None]
        diagonals = torch.eye(entries.size(0), dtype=bool, device=entries.device)[..., None, None]
        x = self.src_encoder(x)[:, None] + self.dst_encoder(x)[None, :] + pairs # [L, L, N, D]
        vec = padded_pos[:, None]  - padded_pos[None, :] # [L, L, N, 3]
        vec_hat = F.normalize(vec, dim=-1)

        dist = torch.linalg.norm(vec, dim=-1, keepdim=True) # [L, L, N, 1]
        dist.masked_fill_(~entries | diagonals, 0)

        # regress outputs
        energy_pairs = self.energy_out(x) # [L, L, N, 1]

        if self.multipole:
            multipoles = self.forces_out(x) # [L, L, N, 4]
            force_pairs = (
                - multipoles[..., :1] * vec_hat / dist.pow(2) # monopole
                - 3 * vec_hat / dist.pow(3) * (multipoles[:, 1:4] * vec_hat).sum(-1, keepdim=True) # dipole
                + multipoles[:, 1:4] / dist.pow(3) # dipole
            )
        else:
            force_magnitudes = self.forces_out(x) # [L, L, N, 1]
            force_pairs = force_magnitudes * vec_hat # [L, L, N, 3]

        force_pairs.masked_fill_(~entries | diagonals, 0)

        energy = (energy_pairs * entries.float()).sum(0).mean(0) # [N, 1]
        forces = (force_pairs * entries.float()).sum(0) # [L, N, 3]
        forces = forces.transpose(0, 1)[padded_node_mask] # [S, 3]

        energy = self.energy_scale * energy
        forces = self.forces_scale * forces

        return energy, forces
        

class PairEmbed(nn.Module):
    def __init__(
        self,
        ff_dim: int = 1024,
        dropout: float = 0,
        num_elements: int = 100,
        num_head = 8,
        num_layers = 12,
        num_gaussians: int = 50,
        rbf_radius: float = 12.,
    ):
        super().__init__()

        offset = torch.linspace(0, rbf_radius, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

        self.dist_encoder = nn.Sequential(
            nn.Linear(num_gaussians, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim)
        )

        self.src_anum_encoder = nn.Embedding(
            num_elements,
            ff_dim,
            padding_idx=0
        )

        self.dst_anum_encoder = nn.Embedding(
            num_elements,
            ff_dim,
            padding_idx=0
        )

        self.attn_encoder = nn.Sequential(
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_layers * num_head)
        )

        self.output_encoder = nn.Sequential(
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim)
        )

        self.scale = nn.Parameter(torch.zeros(num_layers, 1, 1, 1))
        self.num_layers = num_layers
        self.num_heads = num_head

    def forward(
        self,
        padded_pos: torch.Tensor,
        padded_anum: torch.Tensor,
        padded_node_mask: torch.Tensor,
    ):
        # extend to pairs
        entries = padded_node_mask.T[:, None] & padded_node_mask.T[None, :] # [L, L, N]
        diagonals = torch.eye(entries.size(0), dtype=bool, device=entries.device)[..., None]

        # calculate distances
        vec = padded_pos[:, None]  - padded_pos[None, :] # [L, L, N, 3]
        dist = torch.linalg.norm(vec, dim=-1) # [L, L, N]

        # compute rbfs and embeddings
        dist = dist[..., None] - self.offset.view(1, 1, 1, -1) # [L, L, N, G]
        rbf = torch.exp(self.coeff * dist.square())

        # mask out paddings and diagonals
        rbf.masked_fill_((~entries | diagonals)[..., None], 0)
        x = self.dist_encoder(rbf) # [L, L, N, FF]

        # add atomic number embeddings
        x += self.src_anum_encoder(padded_anum)[:, None] + self.dst_anum_encoder(padded_anum)[None, :]

        # project to attn
        attn = self.attn_encoder(x) # [L, L, N, H * layers]
        attn.masked_fill_((~entries | diagonals)[..., None], 0)

        # reshape to fit attn mask shape requirement
        attn = attn.permute(2, 3, 0, 1) # [N, H * layers, L, L]
        attn = attn.reshape(attn.size(0), self.num_layers, self.num_heads, attn.size(2), attn.size(3)) # [N, layers, H, L, L]
        attn = attn.permute(1, 0, 2, 3, 4) # [layers, N, H, L, L]
        attn = attn.reshape(attn.size(0), -1, attn.size(3), attn.size(4)) # [layers, N * H, L, L]
        attn = self.scale * attn
        
        # project into embeddings
        output = self.output_encoder(x)
        output.masked_fill_((~entries | diagonals)[..., None], 0)

        return attn, output
    
class SelfAttentionLayer(nn.Module):
    """
    An attention block implementation based on https://nn.labml.ai/transformers/models.html
    """
    def __init__(
        self, 
        d_model: int = 512, 
        d_ff: int = 2048,
        heads: int = 8, 
        dropout: float = 0,
        activation: nn.Module = nn.GELU()
    ):
        """
        Initialize an `SelfAttentionLayer` instance
        arguments:
            d_model: the size of input 
            d_ff: hidden size of the feed forward network
            heads: number of heads used in MHA
            dropout: dropout strength
            activation: activation function
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.activation = activation

        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None, 
        attn_mask: torch.Tensor = None
    ) -> torch.tensor:
        """
        transform the input using the attention block
        arguments:
            x: input sequence of shape (L, N, C)
            padding_mask: a mask of shape (N, L) with `True` represents padding
        returns:
            transformed sequence of shape (L, N, C)
        """

        z = self.norm_attn(x)
        self_attn, *_ = self.self_attn(z, z, z, key_padding_mask=padding_mask, attn_mask=attn_mask)
        x = x + self.dropout(self_attn)
        x = self.inv_sqrt_2 * x
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        x = self.inv_sqrt_2 * x
        
        return x