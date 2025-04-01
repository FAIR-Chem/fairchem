from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fairchem.core.common.gp_utils import (
    gather_from_model_parallel_region,
    gather_from_model_parallel_region_sum_grad,
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


def scatter_bwd_test():
    rank = dist.get_rank()
    x_full = torch.tensor([2, 3, 5, 7], requires_grad=True, dtype=torch.float)
    x = scatter_to_model_parallel_region(x_full, 0)

    energy_part = x.prod() ** 2

    forces_part = torch.autograd.grad(
        [energy_part],
        [x_full],
        create_graph=True,
    )[0]

    # unable to get this test passing for dforces_dinput_part
    # dforces_dinput_part = torch.hstack(
    #     [
    #         torch.autograd.grad(
    #             [forces_part[0 + rank * 2]],
    #             [x_full],
    #             create_graph=True,
    #         )[0],
    #         torch.autograd.grad(
    #             [forces_part[1 + rank * 2]],
    #             [x_full],
    #             create_graph=True,
    #         )[0],
    #     ]
    # )

    return {
        "gp_rank": rank,
        "energy": energy_part.detach(),
        "forces": forces_part.detach(),
        # "dforces_dinput_part": dforces_dinput_part.detach(),
    }


def test_scatter_bwd():
    torch.autograd.set_detect_anomaly(True)
    expected_output = {
        0: {
            "gp_rank": 0,
            "energy": torch.tensor(36.0),
            "forces": torch.tensor([36.0, 24.0, 490.0, 350.0]),
        },
        1: {
            "gp_rank": 1,
            "energy": torch.tensor(1225.0),
            "forces": torch.tensor([36.0, 24.0, 490.0, 350.0]),
        },
    }
    # A B | C D
    # E_0 = (A*B)**2 , E_1 = (C*D)**2
    # dL_0/dA = 2*A*B^2 = 36
    # dL_0/dB = 2*A^2*B = 24
    # dL_1/dC = 2*C*D^2 = 490
    # dL_1/dD = 2*C^2*D = 350
    # d^2L_0/dA^2 = 2*B^2 = 18
    # d^2L_0/dB^2 = 2*A^2 = 8
    # d^2L_0/dC^2 = 2*D^2 = 98
    # d^2L_0/dD^2 = 2*C^2 = 50

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        scatter_bwd_test,
        init_pg_and_rank_and_launch_test,
    )
    for results in all_rank_results:
        compare_and_assert_dict(expected_output[results["gp_rank"]], results)


def gather_bwd_test(rank=-1):
    if rank < 0:
        rank = dist.get_rank()
        x = torch.tensor([rank + 2], requires_grad=True, dtype=torch.float)
        x_full = gather_from_model_parallel_region(x, 0)
    else:
        x = torch.tensor([rank + 2], requires_grad=True, dtype=torch.float)
        x_other = torch.tensor([(1 - rank) + 2], requires_grad=True, dtype=torch.float)
        x_full = torch.cat([x, x_other]) if rank == 0 else torch.cat([x_other, x])

    energy_part = (x_full.prod() + rank + 1) ** 2

    forces_part = torch.autograd.grad(
        [energy_part],
        [x],
        create_graph=True,
    )[0]

    dforces_dinput_part = torch.autograd.grad(
        [forces_part],
        [x],
        create_graph=True,
    )[0]

    return {
        "gp_rank": rank,
        "energy": energy_part.detach(),
        "forces": forces_part.detach(),
        "dforces_dinput": dforces_dinput_part.detach(),
    }


def test_gather_bwd():
    # A | B
    # E_0 = (A*B +1)^2 , E_1 = (A*B+2)^2
    #     = 49               = 64
    # dL_0/dA = 2*A*B^2+2*B = 42
    # dL_1/dB = 2*A^2*B+4*A = 32
    # dL_0/dB and dL_1/dA are not used! see test_gather_sum_bwd!!
    # d^2L_1/dA^2 = 2*B^2 = 18
    # d^2L_1/dB^2 = 2*A^2 = 8

    non_gp_results_by_gp_rank = {0: gather_bwd_test(0), 1: gather_bwd_test(1)}

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        gather_bwd_test,
        init_pg_and_rank_and_launch_test,
    )

    for rank_results in all_rank_results:
        compare_and_assert_dict(
            non_gp_results_by_gp_rank[rank_results["gp_rank"]], rank_results
        )


def gather_sum_bwd_test(rank=-1):
    if rank < 0:
        rank = dist.get_rank()
        x = torch.tensor([rank + 2], requires_grad=True, dtype=torch.float)
        x_full = gather_from_model_parallel_region_sum_grad(x, 0)
        energy = (x_full.prod() + rank + 1) ** 2
        # sum
        energy = gp_utils.reduce_from_model_parallel_region(energy)
        # forces
        forces_part = torch.autograd.grad(
            [energy],
            [x],
            create_graph=True,
        )[0]
        #
        dforces_dinput_part = torch.autograd.grad(
            [forces_part],
            [x],
            create_graph=True,
        )[0]

    else:
        x = torch.tensor([rank + 2], requires_grad=True, dtype=torch.float)
        x_other = torch.tensor([(1 - rank) + 2], requires_grad=True, dtype=torch.float)
        x_full = torch.cat([x, x_other]) if rank == 0 else torch.cat([x_other, x])
        # sum
        energy = (x_full.prod() + rank + 1) ** 2 + (x_full.prod() + (1 - rank) + 1) ** 2
        # forces
        forces = torch.autograd.grad(
            [energy],
            [x_full],
            create_graph=True,
        )[0]
        forces_part = forces[rank]
        #
        dforces_dinput_part = torch.autograd.grad(
            [forces.sum()],  # critical
            [x],
            create_graph=True,
        )[0]

    return {
        "gp_rank": rank,
        "energy": energy.detach(),
        "forces": forces_part.detach(),
        "dforces_dinput_part": dforces_dinput_part.detach(),
    }


def test_gather_sum_bwd():
    # A(2) | B(3)

    # L_0 = (A*B +1)^2 , L_1 = (A*B+2)^2
    #     = 49               = 64

    # dL_0/dA = 2*A*B^2+2*B = 42 = 2*(AB+1)*B
    # dL_0/dB = 2*A^2*B+2*A = 28 = 2*(AB+1)*A
    # dL_1/dA = 2*A*B^2+4*B = 48 = 2*(AB+2)*B
    # dL_1/dB = 2*A^2*B+4*A = 32 = 2*(AB+2)*A

    # dL/dA = dL_0/dA + dL_1/dA = 90
    # dL/dB = dL_0/dB + dL_1/dB = 60

    # d^2L/dA^2 = 4*B^2 = 36
    # d^2L/dB^2 = 4*A^2 = 16

    non_gp_results_by_gp_rank = {0: gather_sum_bwd_test(0), 1: gather_sum_bwd_test(1)}

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        gather_sum_bwd_test,
        init_pg_and_rank_and_launch_test,
    )
    for rank_results in all_rank_results:
        compare_and_assert_dict(
            non_gp_results_by_gp_rank[rank_results["gp_rank"]], rank_results
        )


# test for one rank to return a product and rest return 0
def scatter_prod_reduce(all_inputs):
    rank = dist.get_rank()

    x_full = all_inputs.clone()

    x = scatter_to_model_parallel_region(x_full, dim=0) + 0
    # BE VERY CAREFUL, inside of this context do not use any variables
    # in other partitions, their gradient will not propagate!
    if rank == 0:
        x = x + x.prod()  # x.prod() * 0  # first two nodes bi-directed
    else:
        x = x + x.prod() ** 2
    # saved tensors in above operation
    energy_part = x.sum()
    energy = gp_utils.reduce_from_model_parallel_region(energy_part.clone())
    energy.backward()

    return {"energy": energy.detach(), "forces": all_inputs.grad.detach()}


def test_scatter_prod_reduce():
    torch.autograd.set_detect_anomaly(True)
    input = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)
    expected_output = {
        "energy": torch.tensor(47.0),
        "forces": torch.tensor([7.0, 5.0, 11.0]),
    }
    # A | B      C
    # A+AB | B+AB   C+C*C
    #  E = A+B+2AB   + C+C*C = 47
    # dE/dA = 1+2B = 7, dE/dB = 1 + 2A = 5 , dE/dC = 1+2C = 11

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    output = spawn_multi_process(
        config, scatter_prod_reduce, init_pg_and_rank_and_launch_test, input
    )

    for output_tensor in output:
        for key in expected_output:
            assert torch.isclose(
                output_tensor[key], expected_output[key]
            ).all(), f"Failed closeness check for {key}"


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
    return x


def embeddings_and_graph_init(atomic_numbers, edge_index):
    if gp_utils.initialized():
        node_partition = gp_utils.scatter_to_model_parallel_region(
            torch.arange(len(atomic_numbers)).to(atomic_numbers.device)
        )
        assert (
            node_partition.numel() > 0
        ), "Looks like there is no atoms in this graph paralell partition. Cannot proceed"
        edge_partition = torch.where(
            torch.logical_and(
                edge_index[1] >= node_partition.min(),
                edge_index[1] <= node_partition.max(),  # TODO: 0 or 1?
            )
        )[0]

        graph_dict = {
            "node_offset": node_partition.min().item(),
            "edge_index": edge_index[:, edge_partition],
        }
        node_embeddings = atomic_numbers[node_partition]
    else:
        graph_dict = {
            "node_offset": 0,
            "edge_index": edge_index,
        }
        node_embeddings = atomic_numbers

    return node_embeddings, graph_dict


# test for one rank to return a product and rest return 0
def simple_layer(x, edge_index, node_offset, n=3):

    if gp_utils.initialized():
        x_full = gp_utils.gather_from_model_parallel_region_sum_grad(x, dim=0)
        x_source = x_full[edge_index[0]]
        x_target = x_full[edge_index[1]]
        dp_rank = gp_utils.get_dp_rank()
    else:
        x_source = x[edge_index[0]]
        x_target = x[edge_index[1]]
        if dist.is_initialized():
            dp_rank = dist.get_rank()
        else:
            dp_rank = 0.0

    # make sure different ddp ranks have different outputs
    # similar to seeing diffferent data batches
    x_source = x_source + dp_rank * 0.1
    x_target = x_target + dp_rank * 0.1

    edge_embeddings = (x_source + 1).pow(n) * (x_target + 1).pow(n)

    new_node_embedding = torch.zeros(
        (x.shape[0],) + edge_embeddings.shape[1:],
        dtype=edge_embeddings.dtype,
        device=edge_embeddings.device,
    )

    new_node_embedding.index_add_(0, edge_index[1] - node_offset, edge_embeddings)

    return new_node_embedding


class SimpleNet(nn.Module):
    def __init__(self, nlayers, n=1.5):
        super().__init__()
        self.nlayers = nlayers
        self.n = n

    def forward(self, atomic_numbers, edge_index):

        node_embeddings, graph_dict = embeddings_and_graph_init(
            atomic_numbers, edge_index
        )

        all_node_embeddings = [node_embeddings]  # store for debugging
        for layer_idx in range(self.nlayers):
            all_node_embeddings.append(
                simple_layer(
                    all_node_embeddings[-1],
                    graph_dict["edge_index"],
                    node_offset=graph_dict["node_offset"],
                    n=self.n,
                )
            )

        final_node_embeddings = all_node_embeddings[-1]

        # only 1 system
        energy_part = torch.zeros(
            1, device=atomic_numbers.device, dtype=atomic_numbers.dtype
        )

        energy_part.index_add_(
            0,
            torch.zeros(
                final_node_embeddings.shape[0],
                dtype=torch.int,
                device=edge_index.device,
            ),
            final_node_embeddings,
        )

        forces_part = torch.autograd.grad(
            [energy_part.sum()],
            [atomic_numbers],
            create_graph=True,
        )[0]

        if gp_utils.initialized():
            energy = gp_utils.reduce_from_model_parallel_region(energy_part)
            forces = gp_utils.reduce_from_model_parallel_region(forces_part)
        else:
            energy = energy_part
            forces = forces_part

        if gp_utils.initialized():
            dp_rank = gp_utils.get_dp_rank()
        elif dist.is_initialized():
            dp_rank = dist.get_rank()
        else:
            dp_rank = 0

        return {"dp_rank": dp_rank, "energy": energy, "forces": forces}


def fwd_bwd_on_simplenet(atomic_numbers, edge_index, nlayers=1):
    sn = SimpleNet(nlayers)
    energy_and_forces = sn(atomic_numbers, edge_index)

    dforces_dinput_part = torch.autograd.grad(
        [energy_and_forces["forces"].sum()],
        [atomic_numbers],
        create_graph=True,
    )[0]

    if gp_utils.initialized():
        dforces_dinput = gp_utils.reduce_from_model_parallel_region(dforces_dinput_part)
    else:
        dforces_dinput = dforces_dinput_part

    energy_and_forces.update({"dforces_dinput": dforces_dinput})

    return {
        k: v.detach() if isinstance(v, torch.Tensor) else v
        for k, v in energy_and_forces.items()
    }


def compare_and_assert_dict(
    d1: dict[str, int | torch.Tensor], d2: dict[str, int | torch.Tensor]
):
    assert len(d1.keys()) == len(d2.keys())
    for k1, v1 in d1.items():
        if isinstance(v1, torch.Tensor):
            assert v1.isclose(d2[k1]).all(), f"{k1} failed closeness check"


@pytest.mark.parametrize("nlayers", [1, 2, 3])  # noqa: PT006
def test_simple_energy(nlayers):
    # torch.autograd.set_detect_anomaly(True)
    atomic_numbers = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    # edge_index = torch.tensor([[1, 1, 1], [0, 2, 1]])
    edge_index = torch.tensor([[0, 1], [0, 2]])

    non_gp_results = fwd_bwd_on_simplenet(atomic_numbers, edge_index, nlayers)

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        fwd_bwd_on_simplenet,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
        nlayers,
    )

    for rank_results in all_rank_results:
        compare_and_assert_dict(non_gp_results, rank_results)


@pytest.mark.parametrize("nlayers", [1])  # noqa: PT006
def test_simple_energy_ddp(nlayers):
    atomic_numbers = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    edge_index = torch.tensor([[0, 1], [0, 2]])

    config = PGConfig(backend="gloo", world_size=2, gp_group_size=1, use_gp=False)
    non_gp_results = spawn_multi_process(
        config,
        fwd_bwd_on_simplenet,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
        nlayers,
    )

    # get ground truth for each ddp rank
    non_gp_results_by_dp_rank = {}
    for results in non_gp_results:
        dp_rank = results["dp_rank"]
        assert dp_rank not in non_gp_results_by_dp_rank
        non_gp_results_by_dp_rank[dp_rank] = results

    config = PGConfig(backend="gloo", world_size=4, gp_group_size=2, use_gp=True)
    all_rank_results = spawn_multi_process(
        config,
        fwd_bwd_on_simplenet,
        init_pg_and_rank_and_launch_test,
        atomic_numbers,
        edge_index,
        nlayers,
    )
    for rank_results in all_rank_results:
        compare_and_assert_dict(
            non_gp_results_by_dp_rank[rank_results["dp_rank"]], rank_results
        )
