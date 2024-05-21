import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ResMLP

class GaussianSmearing(nn.Module):

    def __init__(
        self,
        begin: float = 0,
        stop: float = 1,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(begin, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, x):
        # compute rbfs and embeddings
        x = x[..., None] - self.offset.view(1, 1, 1, -1) # [L, L, N, G]
        rbf = torch.exp(self.coeff * x.square())
        return rbf


class OutputModule(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        dropout: float = 0,
        hidden_layers: int = 3,
        num_gaussians: int = 50,
        max_radius: float = 12.
    ):
        super().__init__()

        self.energy_input = ResMLP(
            input_dim=3*embed_dim,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            num_layers=hidden_layers,
            dropout=dropout
        )

        self.energy_rbf = nn.Sequential(
            GaussianSmearing(0, max_radius, num_gaussians),
            nn.Linear(num_gaussians, embed_dim, bias=False),
        )

        self.energy_output = nn.Linear(embed_dim, 1, bias=False)

        self.forces_input = ResMLP(
            input_dim=3*embed_dim,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            num_layers=hidden_layers,
            dropout=dropout
        )

        self.forces_rbf = nn.Sequential(
            GaussianSmearing(0, max_radius, num_gaussians),
            nn.Linear(num_gaussians, embed_dim, bias=False),
        )

        self.forces_output = nn.Linear(embed_dim, 1, bias=False)

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
        x = torch.cat([
            x[:, None].expand_as(pairs), 
            x[None, :].expand_as(pairs), 
            pairs
        ], dim=-1) # [L, L, N, 3D]
        vec = padded_pos[:, None]  - padded_pos[None, :] # [L, L, N, 3]
        vec_hat = F.normalize(vec, dim=-1)
        dist = torch.linalg.norm(vec, dim=-1) # [L, L, N]

        # regress outputs
        energy_pairs = self.energy_input(x) * self.energy_rbf(dist)
        force_pairs = self.forces_input(x) * self.forces_rbf(dist)
        energy_pairs = self.energy_output(energy_pairs)
        force_pairs = self.forces_output(force_pairs) * vec_hat # [L, L, N, 3]

        energy = (energy_pairs * entries.float()).sum(0).mean(0) # [N, 1]
        forces = (force_pairs * entries.float()).sum(0) # [L, N, 3]
        forces = forces.transpose(0, 1)[padded_node_mask] # [S, 3]

        return energy, forces
        

class PairEmbed(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        dropout: float = 0,
        num_heads = 8,
        num_layers = 12,
        num_gaussians: int = 50,
        rbf_radius: float = 12.,
    ):
        super().__init__()

        offset = torch.linspace(0, rbf_radius, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

        self.dist_encoder = ResMLP(
            input_dim=num_gaussians,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            dropout=dropout
        )

        self.attn_encoder = ResMLP(
            input_dim=embed_dim,
            hidden_dim=ff_dim,
            output_dim=num_layers*num_heads,
            dropout=dropout
        )

        self.output_encoder = ResMLP(
            input_dim=embed_dim,
            hidden_dim=ff_dim,
            output_dim=embed_dim,
            dropout=dropout
        )

        self.rbf = GaussianSmearing(0, rbf_radius, num_gaussians=num_gaussians)

        self.num_layers = num_layers
        self.num_heads = num_heads

    def forward(
        self,
        x: torch.Tensor,
        padded_pos: torch.Tensor,
        padded_node_mask: torch.Tensor,
    ):
        # extend to pairs
        entries = padded_node_mask.T[:, None] & padded_node_mask.T[None, :] # [L, L, N]
        diagonals = torch.eye(entries.size(0), dtype=bool, device=entries.device)[..., None]

        # calculate distances
        vec = padded_pos[:, None] - padded_pos[None, :] # [L, L, N, 3]
        dist = torch.linalg.norm(vec, dim=-1) # [L, L, N]

        # compute rbfs and embeddings
        rbf = self.rbf(dist)

        # add embeddings
        x = x[:, None] + x[None, :] + self.dist_encoder(rbf) # [L, L, N, FF]

        # project to attn
        attn = self.attn_encoder(x) # [L, L, N, H * layers]
        attn.masked_fill_((~entries)[..., None], -torch.inf)

        # reshape to fit attn mask shape requirement
        attn = attn.permute(2, 3, 0, 1) # [N, H * layers, L, L]
        attn = attn.reshape(attn.size(0), self.num_layers, self.num_heads, attn.size(2), attn.size(3)) # [N, layers, H, L, L]
        attn = attn.permute(1, 0, 2, 3, 4) # [layers, N, H, L, L]
        attn = attn.reshape(attn.size(0), -1, attn.size(3), attn.size(4)) # [layers, N * H, L, L]
        
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
        self.feed_forward = ResMLP(
            input_dim=d_model,
            hidden_dim=d_ff,
            output_dim=d_model,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.activation = activation

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
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + self.dropout(ff)
        
        return x