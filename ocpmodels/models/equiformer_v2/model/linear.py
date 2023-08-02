import torch
import math


def Linear_gaussian_init(in_channels, out_channels, bias=True):

    lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
    std = 1 / math.sqrt(in_channels)
    torch.nn.init.normal_(lin.weight, 0, std)

    if bias:
        torch.nn.init.constant_(lin.bias, 0)

    return lin