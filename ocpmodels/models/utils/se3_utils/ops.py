import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def shape_is(a, b, ignore_batch=1):
    """
    check whether multi-dimensional array a has dimensions b; use in combination with assert

    :param a: multi dimensional array
    :param b: list of ints which indicate expected dimensions of a
    :param ignore_batch: if set to True, ignore first dimension of a
    :return: True or False
    """
    if ignore_batch:
        shape_a = np.array(a.shape[1:])
    else:
        shape_a = np.array(a.shape)
    shape_b = np.array(b)
    return np.array_equal(shape_a, shape_b)


def norm_with_epsilon(input_tensor, axis=None, keep_dims=False, epsilon=0.0):
    """
    Regularized norm

    Args:
        input_tensor: torch.Tensor

    Returns:
        torch.Tensor normed over axis
    """
    # return torch.sqrt(torch.max(torch.reduce_sum(torch.square(input_tensor), axis=axis, keep_dims=keep_dims), epsilon))
    keep_dims = bool(keep_dims)
    squares = torch.sum(input_tensor ** 2, axis=axis, keepdim=keep_dims)
    squares = torch.max(squares, torch.tensor([epsilon]).to(squares.device))
    return torch.sqrt(squares)
