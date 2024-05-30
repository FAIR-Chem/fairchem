import torch
import torch.nn as nn

from .mlp import ResMLP
from .rbf import RadialBasisFunction
    
class PairEmbed(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_masks: int = 1,
        num_gaussians: int = 50,
        rbf_radius: float = 12.,
        trainable_rbf: bool = False,
        dropout: float = 0.,
        num_layers: int = 2,
        use_gated_mlp: bool = False,
        sparse: bool = False
    ):
        super().__init__()

        self.use_gated_mlp = use_gated_mlp
        self.num_heads = num_heads
        self.num_masks = num_masks
        self.sparse = sparse
        self.rbf_radius = rbf_radius

        self.mlp = ResMLP(
            input_dim=3*embed_dim,
            hidden_dim=hidden_dim,
            output_dim=num_heads*num_masks,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.rbf = RadialBasisFunction(
            rbf_radius=rbf_radius,
            num_gaussians=num_gaussians,
            embed_dim=embed_dim,
            trainable=trainable_rbf
        )

        if use_gated_mlp:
            self.gate_rbf = RadialBasisFunction(
                rbf_radius=rbf_radius,
                num_gaussians=num_gaussians,
                embed_dim=hidden_dim,
                trainable=trainable_rbf
            )
    
    def forward(
        self,
        x: torch.Tensor,
        dist: torch.Tensor,
        mask: torch.Tensor,
    ):
        # prepare pairs    
        pairs = self.rbf(dist)
        entries = mask.T[None, :] & mask.T[:, None]
        if self.sparse:
            entries &= dist < self.rbf_radius
        entries = entries[..., None]

        # prepare inputs
        L, B, D = x.shape
        inputs = torch.cat([
            x[None, :].expand(L, L, B, D),
            x[:, None].expand(L, L, B, D),
            pairs
        ], dim = -1)

        # get attention bias of shape [L, L, N, H * M]
        if self.use_gated_mlp:
            attn_bias = self.mlp(inputs, gate=self.gate_rbf(dist))
        else:
            attn_bias = self.mlp(inputs)
        attn_bias.masked_fill_(~entries, -torch.inf)

        # get attention bias of shape [M, N * H, L, L]
        attn_bias = attn_bias.reshape(L, L, B, self.num_heads, self.num_masks)
        attn_bias = attn_bias.permute(4, 2, 3, 0, 1)
        attn_bias = attn_bias.reshape(self.num_masks, B * self.num_heads, L, L)

        return attn_bias