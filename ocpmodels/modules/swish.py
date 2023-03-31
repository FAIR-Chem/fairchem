import torch


def swish(x):
    return torch.nn.functional.silu(x)


class Swish(torch.nn.SiLU):
    pass
