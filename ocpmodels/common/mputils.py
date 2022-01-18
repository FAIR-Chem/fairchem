import math
import torch
from torch import distributed as dist
from typing import Any, List


########## INITIALIZATION ##########

_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None


def ensure_div(a, b):
    assert a % b == 0


def divide_and_check_no_remainder(a: int, b: int):
    ensure_div(a, b)
    return a // b


def setup_mp(mp_size, backend):
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()

    mp_size = min(mp_size, world_size)
    ensure_div(world_size, mp_size)
    dp_size = world_size // mp_size
    rank = dist.get_rank()

    if rank == 0:
        print("> initializing model parallel with size {}".format(mp_size))
        print("> initializing ddp with size {}".format(dp_size))

    groups = torch.arange(world_size).reshape(dp_size, mp_size)
    found = [x.item() for x in torch.where(groups == rank)]

    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    for j in range(mp_size):
        group = dist.new_group(groups[:, j].tolist(), backend=backend)
        if j == found[1]:
            _DATA_PARALLEL_GROUP = group
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for i in range(dp_size):
        group = dist.new_group(groups[i, :].tolist(), backend=backend)
        if i == found[0]:
            _MODEL_PARALLEL_GROUP = group


def cleanup_mp():
    dist.destroy_process_group(_DATA_PARALLEL_GROUP)
    dist.destroy_process_group(_MODEL_PARALLEL_GROUP)


def initialized():
    return _MODEL_PARALLEL_GROUP is not None


def get_dp_group():
    return _DATA_PARALLEL_GROUP


def get_mp_group():
    return _MODEL_PARALLEL_GROUP


def get_dp_rank():
    return dist.get_rank(group=get_dp_group())


def get_mp_rank():
    return dist.get_rank(group=get_mp_group())


def get_dp_world_size():
    return dist.get_world_size(group=get_dp_group())


def get_mp_world_size():
    return dist.get_world_size(group=get_mp_group())


########## DIST METHODS ##########

def pad_tensor(tensor: torch.Tensor, dim: int = -1):
    size = tensor.size(dim)
    world_size = get_mp_world_size()
    if size % world_size == 0:
        return tensor
    else:
        pad_size = world_size - size % world_size
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        padding = torch.empty(pad_shape, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=dim)


def trim_tensor(tensor: torch.Tensor, sizes: torch.Tensor = None, dim: int = 0):
    size = tensor.size(dim)
    world_size = get_mp_world_size()
    if size % world_size == 0:
        return tensor, sizes
    trim_size = size - size % world_size
    if dim == 0:
        tensor = tensor[:trim_size]
    elif dim == 1:
        tensor = tensor[:, :trim_size]
    else:
        raise ValueError
    if sizes is not None:
        sizes[-1] = sizes[-1] - size % world_size
    return tensor, sizes
    

def _split_tensor(tensor: torch.Tensor, num_parts: int, dim: int = -1, contiguous_chunks: bool = False):
    # part_size = divide_and_check_no_remainder(tensor.size(-1), num_parts)
    part_size = math.ceil(tensor.size(dim) / num_parts)
    tensor_list = torch.split(tensor, part_size, dim=dim)
    if contiguous_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def _reduce(ctx: Any, input: torch.Tensor) -> torch.Tensor:
    group = get_mp_group()
    if ctx:
        ctx.mark_dirty(input)
    if dist.get_world_size(group) == 1:
        return input
    dist.all_reduce(input, group=group)
    return input


def _split(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    group = get_mp_group()
    rank = get_mp_rank()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input
    input_list = _split_tensor(input, world_size, dim=dim)
    return input_list[rank].contiguous()


def _gather(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    group = get_mp_group()
    rank = get_mp_rank()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input
    tensor_list = [torch.empty_like(input) for _ in range(world_size)]
    tensor_list[rank] = input
    dist.all_gather(tensor_list, input, group=group)
    return torch.cat(tensor_list, dim=dim).contiguous()


class CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _reduce(None, grad_output)


class ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        return _reduce(ctx, input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class ScatterToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1):
        result = _split(input, dim)
        ctx.save_for_backward(torch.tensor(dim))
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dim, = ctx.saved_tensors
        return _gather(grad_output, dim.item()), None


class GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1):
        ctx.save_for_backward(torch.tensor(dim))
        return _gather(input, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dim, = ctx.saved_tensors
        result = _split(grad_output, dim.item())
        return result, None


def copy_to_model_parallel_region(input: torch.Tensor) -> torch.Tensor:
    return CopyToModelParallelRegion.apply(input)


def reduce_from_model_parallel_region(input: torch.Tensor) -> torch.Tensor:
    return ReduceFromModelParallelRegion.apply(input)


def scatter_to_model_parallel_region(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return ScatterToModelParallelRegion.apply(input, dim)


def gather_from_model_parallel_region(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return GatherFromModelParallelRegion.apply(input, dim)
