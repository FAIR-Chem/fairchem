import torch
import torch.nn as nn

from torch.nn import Linear
from .activation import (
    ScaledSiLU,
    ScaledSwiGLU,
    SwiGLU
)


class EnergyBlock(torch.nn.Module):
    """
    Energy Block: Output block computing the energy

    Args:
        in_channels (int):          Number of input channels
        hidden_channels (int):      Number of hidden channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        activation (str):           Name of non-linear activation function
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_sphere_samples,
        activation='scaled_silu'
    ):
        super(EnergyBlock, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_sphere_samples = num_sphere_samples
        self.activation = activation

        # same as `S2ActivationFeedForwardNetwork`
        if activation in ['scaled_silu', 'silu']:
            if activation == 'scaled_silu':
                act = ScaledSiLU
            elif activation == 'silu':
                act = nn.SiLU
            else:
                raise ValueError
            self.mlp_pt = nn.Sequential(
                nn.Linear(self.in_channels, self.hidden_channels, bias=True),
                act(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
                act(),
                nn.Linear(self.hidden_channels, 1, bias=True)
            )
        elif activation in ['swiglu']:
            self.mlp_pt = nn.Sequential(
                SwiGLU(self.in_channels, self.hidden_channels, bias=True),
                #SwiGLU(self.hidden_channels, self.hidden_channels, bias=True),
                nn.Linear(self.hidden_channels, 1, bias=True)
            )
        else:
            raise ValueError


    def forward(self, x_l0, x_pt):
        # `x_pt` are the values of the channels sampled at different points on the sphere
        x_pt = self.mlp_pt(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        x_pt = torch.sum(x_pt, dim=1) / self.num_sphere_samples
        out = x_pt

        return out


class EnergyBlockV2(torch.nn.Module):
    """
    Energy Block: Output block computing the energy

    Args:
        in_channels (int):          Number of input channels
        hidden_channels (int):      Number of hidden channels
        activation (str):           Name of non-linear activation function
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        activation='scaled_silu'
    ):
        super(EnergyBlockV2, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.activation = activation

        # same as `S2ActivationFeedForwardNetwork`
        if activation in ['scaled_silu', 'silu']:
            if activation == 'scaled_silu':
                act = ScaledSiLU
            elif activation == 'silu':
                act = nn.SiLU
            else:
                raise ValueError
            self.mlp = nn.Sequential(
                nn.Linear(self.in_channels, self.hidden_channels, bias=True),
                act(),
                #nn.Linear(self.hidden_channels, self.hidden_channels, bias=True),
                #act(),
                nn.Linear(self.hidden_channels, 1, bias=True)
            )
        elif activation in ['swiglu']:
            self.mlp = nn.Sequential(
                SwiGLU(self.in_channels, self.hidden_channels, bias=True),
                #SwiGLU(self.hidden_channels, self.hidden_channels, bias=True),
                nn.Linear(self.hidden_channels, 1, bias=True)
            )
        else:
            raise ValueError


    def forward(self, x):
        out = self.mlp(x)
        return out


class ForceBlock(torch.nn.Module):
    """
    Force Block: Output block computing the per atom forces

    Args:
        in_channels (int):          Number of input channels
        hidden_channels (int):      Number of hidden channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        activation (str):           Name of non-linear activation function
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_sphere_samples,
        activation='scaled_silu'
    ):
        super(ForceBlock, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_sphere_samples = num_sphere_samples
        self.activation = activation

        # same as `S2ActivationFeedForwardNetwork`
        if activation in ['scaled_silu', 'silu']:
            if activation == 'scaled_silu':
                act = ScaledSiLU
            elif activation == 'silu':
                act = nn.SiLU
            else:
                raise ValueError
            self.module = nn.Sequential(
                nn.Linear(self.in_channels, self.hidden_channels),
                act(),
                #nn.Linear(self.hidden_channels, self.hidden_channels),
                #act(),
                nn.Linear(self.hidden_channels, 1, bias=False)
            )
        else:
            raise ValueError


    def forward(self, x_pt, sphere_points):
        # `x_pt` are the values of the channels sampled at different points on the sphere
        x_pt = self.module(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        out = x_pt * sphere_points.view(1, self.num_sphere_samples, 3)
        out = torch.sum(out, dim=1) / self.num_sphere_samples
        return out
