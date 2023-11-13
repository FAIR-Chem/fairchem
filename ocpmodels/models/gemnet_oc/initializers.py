"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial

import torch


def _standardize(kernel):
    """
    Makes sure that N*Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He (Kaiming) initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor


def grid_init(tensor, start=-1, end=1):
    """
    Generate a weight matrix so that each input value corresponds to one value on a regular grid between start and end.
    """
    fan_in = tensor.shape[1]

    with torch.no_grad():
        data = torch.linspace(
            start, end, fan_in, device=tensor.device, dtype=tensor.dtype
        ).expand_as(tensor)
        tensor.copy_(data)

    return tensor


def log_grid_init(tensor, start=-4, end=0):
    """
    Generate a weight matrix so that each input value corresponds to one value on a regular logarithmic grid between 10^start and 10^end.
    """
    fan_in = tensor.shape[1]

    with torch.no_grad():
        data = torch.logspace(
            start, end, fan_in, device=tensor.device, dtype=tensor.dtype
        ).expand_as(tensor)
        tensor.copy_(data)

    return tensor


def get_initializer(name, **init_kwargs):
    name = name.lower()
    if name == "heorthogonal":
        initializer = he_orthogonal_init
    elif name == "zeros":
        initializer = torch.nn.init.zeros_
    elif name == "grid":
        initializer = grid_init
    elif name == "loggrid":
        initializer = log_grid_init
    else:
        raise UserWarning(f"Unknown initializer: {name}")

    initializer = partial(initializer, **init_kwargs)
    return initializer
