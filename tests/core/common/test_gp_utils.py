from __future__ import annotations
from functools import partial

import pytest
import torch

from fairchem.core.common.gp_utils import (
    gather_from_model_parallel_region,
    scatter_to_model_parallel_region,
)
from fairchem.core.common.test_utils import (
    PGConfig,
    init_pg_and_rank_and_launch_test,
    spawn_multi_process,
)

from torch import distributed as dist

from fairchem.core.common import gp_utils


def _dummy_call(x):
    return x


@pytest.mark.parametrize(
    "world_size, input, expected_output", [(1, 5, [5]), (3, 0, [0, 0, 0])]
)  # noqa: PT006
def test_basic_setup(world_size: int, input: torch.Tensor, expected_output: list):
    config = PGConfig(
        backend="gloo", world_size=world_size, gp_group_size=1, use_gp=True
    )
    output = spawn_multi_process(
        config, _dummy_call, init_pg_and_rank_and_launch_test, input
    )
    assert output == expected_output


@pytest.mark.parametrize(
    "world_size, gp_size, input, expected_output",  # noqa: PT006
    [
        (
            2,
            1,
            torch.Tensor([0, 1, 2, 3]),
            [torch.Tensor([0, 1, 2, 3]), torch.Tensor([0, 1, 2, 3])],
        ),
        (
            2,
            2,
            torch.Tensor([0, 1, 2, 3]),
            [torch.Tensor([0, 1]), torch.Tensor([2, 3])],
        ),
        (2, 2, torch.Tensor([0, 1, 2]), [torch.Tensor([0, 1]), torch.Tensor([2])]),
        (
            3,
            3,
            torch.Tensor([0, 1, 2]),
            [torch.Tensor([0]), torch.Tensor([1]), torch.Tensor([2])],
        ),
    ],
)
def test_scatter_tensors(
    world_size: int, gp_size: int, input: torch.Tesnor, expected_output: list
):
    config = PGConfig(
        backend="gloo", world_size=world_size, gp_group_size=gp_size, use_gp=True
    )
    output = spawn_multi_process(
        config,
        scatter_to_model_parallel_region,
        init_pg_and_rank_and_launch_test,
        input,
    )
    for out, expected_out in zip(output, expected_output):
        assert torch.equal(out, expected_out)


def scatter_gather_fn(input: torch.Tensor, dim: int = 0):
    x = scatter_to_model_parallel_region(input, dim)
    return gather_from_model_parallel_region(x, dim)


@pytest.mark.parametrize(
    "world_size, gp_size, input, expected_output",  # noqa: PT006
    [
        (
            2,
            1,
            torch.Tensor([0, 1, 2, 3]),
            [torch.Tensor([0, 1, 2, 3]), torch.Tensor([0, 1, 2, 3])],
        ),
        (
            2,
            2,
            torch.Tensor([0, 1, 2, 3]),
            [torch.Tensor([0, 1, 2, 3]), torch.Tensor([0, 1, 2, 3])],
        ),
        (
            2,
            2,
            torch.Tensor([0, 1, 2]),
            [torch.Tensor([0, 1, 2]), torch.Tensor([0, 1, 2])],
        ),
        (
            3,
            3,
            torch.Tensor([0, 1, 2]),
            [torch.Tensor([0, 1, 2]), torch.Tensor([0, 1, 2]), torch.Tensor([0, 1, 2])],
        ),
    ],
)
def test_gather_tensors(
    world_size: int, gp_size: int, input: torch.Tesnor, expected_output: list
):
    config = PGConfig(
        backend="gloo", world_size=world_size, gp_group_size=gp_size, use_gp=True
    )
    output = spawn_multi_process(
        config, scatter_gather_fn, init_pg_and_rank_and_launch_test, input
    )
    for out, expected_out in zip(output, expected_output):
        assert torch.equal(out, expected_out)


# test for one rank to return a product and rest return 0
def gather_prod_rank(all_inputs, target_rank=0):
    rank = dist.get_rank()
    x = scatter_to_model_parallel_region(all_inputs) + 0
    x_full = gather_from_model_parallel_region(x, 0)
    if rank == target_rank:
        loss = x_full.prod()
    else:
        loss = x_full.sum() * 0
    # adding 0.0 makes it out of place for reduce with respect to
    # saved tensors in above operation
    print("RANK", rank, loss, x_full, x)
    loss = gp_utils.reduce_from_model_parallel_region(loss + 0.0)
    loss.backward()
    return all_inputs.grad


def layer(x, target_rank):
    rank = dist.get_rank()

    x_full = gather_from_model_parallel_region(x, 0)
    x_prod = x_full.prod()
    # backward graphs need to be same operation wise
    # otherwise might miss a dist sync
    if rank == target_rank:
        x = x * 0 + x_prod
    else:
        x = x * 0 + x_prod * 0.0 + (rank + 1)
    print("LAYERxx", x_full, rank, x)
    return x


# test for one rank to return a product and rest return 0
def gather_prod_rank_multiple_layer(all_inputs):
    x = scatter_to_model_parallel_region(all_inputs) + 0

    l0 = layer(x, 0)  # ABC, ABC | 2
    print("l0INSP", dist.get_rank(), l0)
    l1 = layer(l0, 1)  # 1,1 | 2*ABC**2
    print("l1INSP", dist.get_rank(), l0)

    l0.retain_grad()
    l1.retain_grad()

    loss = l1.sum()
    # adding 0.0 makes it out of place for reduce with respect to
    # saved tensors in above operation
    loss = gp_utils.reduce_from_model_parallel_region(loss + 0.0)
    loss.backward(retain_graph=True)
    print("l0/l1 grad", dist.get_rank(), l0, l0.grad, "l1", l1, l1.grad)
    return all_inputs.grad


def test_gather_fwd_bwd_multilayer():
    torch.autograd.set_detect_anomaly(True)
    input = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    # expected_output = torch.tensor([6.0, 3.0, 2.0])
    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    output = spawn_multi_process(
        config,
        gather_prod_rank_multiple_layer,
        init_pg_and_rank_and_launch_test,
        input,
    )
    print("OUTPUT", output)

    # for output_tensor in output:
    #    assert torch.isclose(output_tensor, expected_output).all()


def test_gather_fwd_bwd():
    torch.autograd.set_detect_anomaly(True)
    input = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    expected_output = torch.tensor([6.0, 3.0, 2.0])
    for target_rank in [0]:
        config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
        output = spawn_multi_process(
            config,
            partial(gather_prod_rank, target_rank=target_rank),
            init_pg_and_rank_and_launch_test,
            input,
        )
        for output_tensor in output:
            assert torch.isclose(output_tensor, expected_output).all()


# test for one rank to return a product and rest return 0
def gather_prod_rank_grad(all_inputs, target_rank=0):

    from torchviz import make_dot

    rank = dist.get_rank()
    x = scatter_to_model_parallel_region(all_inputs, dim=0).clone()
    # if rank == 0:
    #    x = all_inputs[:2]
    # else:
    #    x = all_inputs[2:]
    # x_full = x
    x_full = gather_from_model_parallel_region(x, 0)
    # x = None
    # x_full = all_inputs + 0.0
    loss = x_full.prod()  # ** 2
    print("x,x_full,rank,loss", x, x_full, rank, loss)
    # adding 0.0 makes it out of place for reduce with respect to
    # saved tensors in above operation
    loss.retain_grad()
    energy = loss  # gp_utils.reduce_from_model_parallel_region(loss)
    energy.retain_grad()
    grads = torch.autograd.grad(
        [energy],
        [all_inputs],
        create_graph=True,
    )
    print("denergy/dinput", rank, grads)
    forces_1d = grads[0]  # gp_utils.reduce_from_model_parallel_region(grads[0])
    forces_1d.retain_grad()
    print("Forces 1d", rank, forces_1d)
    forces_loss = forces_1d.sum()
    forces_loss.retain_grad()

    # grads = torch.autograd.grad(
    #     [forces_loss],
    #     [all_inputs],
    #     create_graph=True,
    # )

    def print_grad_hook(grad):
        print(f"Gradient: {grad}")

    x.register_hook(print_grad_hook)
    x_full.register_hook(print_grad_hook)
    loss.register_hook(print_grad_hook)
    energy.register_hook(print_grad_hook)
    forces_1d.register_hook(print_grad_hook)
    forces_loss.register_hook(print_grad_hook)
    print("force_loss", rank, forces_loss)
    forces_loss.backward(retain_graph=True)
    print("all_inputs.grad", rank, all_inputs.grad)
    print("forces1d_grads, ", forces_1d.grad)
    print("forces_loss grad, ", forces_loss.grad)
    print("energy grad ", energy.grad)
    print("loss grad", loss.grad)
    params = {
        "all_inputs": all_inputs,
        "x": x,
        "x_full": x_full,
        "loss": loss,
        "energy": energy,
        "forces_1d": forces_1d,
        "forces_loss": forces_loss,
    }
    dot = make_dot(
        (x, x_full, energy, forces_1d, forces_loss),
        params=params,
        show_attrs=True,
        show_saved=True,
    )
    dot.render(filename=f"model_graph_{rank}.png", format="png")
    # loss.backward()
    return  # grads.detach()


def test_gather_fwd_bwd_bwd():
    # torch.autograd.set_detect_anomaly(True)
    input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    expected_output = torch.tensor([6.0, 3.0, 2.0])
    for target_rank in [0]:
        config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
        output = spawn_multi_process(
            config,
            partial(gather_prod_rank_grad, target_rank=target_rank),
            init_pg_and_rank_and_launch_test,
            input,
        )
        # for output_tensor in output:
        #    assert torch.isclose(output_tensor, expected_output).all()
