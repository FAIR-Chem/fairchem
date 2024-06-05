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

from .attn import SelfAttentionLayer
from .output import OutputModule
from .pair_embed import PairEmbed
from .mlp import ResMLP
from .rbf import GaussianSmearing

logger = logging.getLogger(__name__)

@registry.register_model("transformer")
class Transformer(BaseModel):
    """
    Pure Transformer

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
    num_gaussians: int
        number of gaussians to use in rbf
    trainable_rbf: bool
        use trainable RBFs
    pair_embed_style: string
        can be "update", "fixed", "shared", or "none", decide the style
        of edge embedding
    gate_pair_embed: bool
        to gate the mlp with radial basis function or not
    sparse: bool
        to use sparse attention or not
    output_layers: int
        number of output layers to use
    gate_output: bool
        to gate output with radial basis function or not
    avg_atoms: float
        average number of atoms
    pos_emb_style: str
        can be "none", "naive", "laplacian"
    pos_scale: float
        the scale of position to normalize by
    lap_heads: int
        number of laplacian heads
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
            otf_graph: bool = False,
            rbf_radius: float = 5.0,
            num_gaussians: int = 50,
            trainable_rbf: bool = False,
            num_pair_embed_layers: int = 2,
            pair_embed_style: str = "update",
            gate_pair_embed: bool = False,
            sparse: bool = False,
            output_layers: int = 3,
            gate_output: bool = False,
            avg_atoms: float = 60,
            pos_emb_style: str = "naive",
            pos_scale: float = None,
            lap_heads: int = None,
        ):

        super().__init__()

        assert pair_embed_style in ["none", "fixed", "shared", "update"]
        assert pos_emb_style in ["none", "naive", "laplacian"]

        self.otf_graph=otf_graph
        self.pos_emb_style = pos_emb_style
        self.pos_scale = pos_scale
        self.pair_embed_style = pair_embed_style
        self.num_layers = num_layers

        if pos_emb_style == "naive":
            self.pos_encoder=ResMLP(
                input_dim=3,
                hidden_dim=hidden_dim,
                output_dim=embed_dim,
                dropout=dropout
            )

        elif pos_emb_style == "laplacian":

            self.lap_k = embed_dim // lap_heads
            self.register_buffer(
                "neg_log_var",
                - 2 * torch.linspace(0, rbf_radius, lap_heads + 1)[1:].log()
            )

            self.pos_encoder = ResMLP(
                input_dim=self.lap_k * lap_heads,
                hidden_dim=hidden_dim,
                output_dim=embed_dim,
                dropout=dropout
            )

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

        if pair_embed_style in ["update", "fixed"]:
            self.pair_embeds =  nn.ModuleList([
                PairEmbed(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_gaussians=num_gaussians,
                    rbf_radius=rbf_radius,
                    trainable_rbf=trainable_rbf,
                    dropout=dropout,
                    num_layers=num_pair_embed_layers,
                    use_gated_mlp=gate_pair_embed,
                    sparse=sparse,
                ) for _ in range(num_layers)
            ])
        elif pair_embed_style == "shared":
            self.pair_embeds = PairEmbed(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_masks=num_layers,
                num_gaussians=num_gaussians,
                rbf_radius=rbf_radius,
                trainable_rbf=trainable_rbf,
                dropout=dropout,
                num_layers=num_pair_embed_layers,
                use_gated_mlp=gate_pair_embed,
                sparse=sparse,
            )

        self.layers = nn.ModuleList([
            SelfAttentionLayer(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])

        self.init_output = OutputModule(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            hidden_layers=output_layers,
            num_gaussians=num_gaussians,
            rbf_radius=rbf_radius,
            trainable_rbf=trainable_rbf,
            use_gated_mlp=gate_output,
            avg_len=avg_atoms,
            sparse=sparse,
        )

        self.outputs = nn.ModuleList([
            OutputModule(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                hidden_layers=output_layers,
                num_gaussians=num_gaussians,
                rbf_radius=rbf_radius,
                trainable_rbf=trainable_rbf,
                use_gated_mlp=gate_output,
                avg_len=avg_atoms,
                sparse=sparse
            ) for _ in range(num_layers)
        ])        

    def tokenize(
            self,
            pos: torch.Tensor,
            natoms: torch.Tensor,
            atomic_numbers: torch.Tensor
        ):

        # tokenize the sequence
        nmax = natoms.max()
        batch_size = len(natoms)
        token_pos = torch.arange(nmax, device=natoms.device)[None, :].repeat(batch_size, 1)
        mask = token_pos < natoms[:, None]

        padded_pos = torch.zeros((batch_size, nmax, 3), device=pos.device, dtype=torch.float)
        padded_pos[mask] = pos
        padded_pos = padded_pos.transpose(0, 1).contiguous()

        assert (self.atomic_number_mask[atomic_numbers] != -1).all()

        padded_anum = torch.zeros((batch_size, nmax), device=natoms.device, dtype=torch.long)
        padded_anum[mask] = self.atomic_number_mask[atomic_numbers]
        padded_anum = padded_anum.transpose(0, 1).contiguous()

        # calculate pairwise features
        vec = padded_pos[:, None]  - padded_pos[None, :] # [L, L, N, 3]
        vec_hat = F.normalize(vec, dim=-1)
        dist = torch.linalg.norm(vec, dim=-1) # [L, L, N]
        entries = mask.T[:, None] & mask.T[None, :]

        # encode the sequence
        if self.pos_emb_style == "naive":
            norm_pos = padded_pos / self.pos_scale
            features = self.pos_encoder(norm_pos) + self.atomic_number_encoder(padded_anum)
        elif self.pos_emb_style == "laplacian":
            # compute laplacian
            log_A = - 0.5 * dist.square()[..., None] * self.neg_log_var.exp()
            I = torch.eye(log_A.size(0), device=log_A.device)[..., None, None].expand_as(log_A)
            log_A[I.bool() | ~entries[..., None]] = - torch.inf
            L = I - torch.exp(0.5 * (F.log_softmax(log_A, dim=0) + F.log_softmax(log_A, dim=1)))
            L.masked_fill_(~entries[..., None], 0)
            L = L.permute(2, 3, 0, 1) # [N, H, L, L]
            # compute eigenvectors
            _, ein_vec = torch.linalg.eigh(L) # [N, H, L, L]
            ein_vec = ein_vec[:, :, :, :self.lap_k] # [N, H, L, d_k]
            ein_vec = ein_vec.permute(2, 0, 1, 3) # [L, N, H, d_k]
            ein_vec = ein_vec.reshape(ein_vec.size(0), ein_vec.size(1), -1) # [L, N, H * d_k]
            # embed eingenvectors
            features = self.pos_encoder(ein_vec) + self.atomic_number_encoder(padded_anum)
        else:
            features = self.atomic_number_encoder(padded_anum)
            
        key_padding_mask = torch.zeros_like(mask, dtype=torch.float)
        key_padding_mask.masked_fill_(~mask, -torch.inf)

        return features, mask, key_padding_mask, vec_hat, dist

    def forward(self, data):

        # extract data
        pos = data.pos
        natoms = data.natoms
        atomic_numbers = data.atomic_numbers.long()

        # tokenize implicit batch
        features, mask, key_padding_mask, vec_hat, dist = self.tokenize(
            pos=pos, natoms=natoms, atomic_numbers=atomic_numbers
        )
        x = features

        # initialize output
        energy, forces = self.init_output(x, dist, vec_hat, mask)

        if self.pair_embed_style == "shared":
            attn_masks = self.pair_embeds(x, dist, mask)

        # forward passing
        for i in range(self.num_layers):
            
            # build attention masks
            if self.pair_embed_style == "update":
                attn_mask = self.pair_embeds[i](x, dist, mask)[0]
            elif self.pair_embed_style == "fixed":
                attn_mask = self.pair_embeds[i](features, dist, mask)[0]
            elif self.pair_embed_style == "shared":
                attn_mask = attn_masks[i]
            else:
                attn_mask = None

            # attention block
            x = self.layers[i](x, key_padding_mask, attn_mask)

            # residual outputs
            e, f = self.outputs[i](x, dist, vec_hat, mask)
            energy += e
            forces += f

        energy = energy / (self.num_layers + 1)
        forces = forces / (self.num_layers + 1)

        return {"energy": energy, "forces": forces}
    
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
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm, GaussianSmearing)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)