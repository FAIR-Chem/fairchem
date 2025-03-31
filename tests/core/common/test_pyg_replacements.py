pyg_available = False
try:
    from torch_scatter import segment_coo, segment_csr

    pyg_available = True
except ImportError:
    pass

import torch
import pytest

from fairchem.core.common.utils import sum_partitions


@pytest.mark.skipif(
    not pyg_available, reason="Pytorch Geometric and libs not installed"
)
def test_segment_coo():
    torch.manual_seed(0)

    n = 10
    for target_size in torch.randint(1, 1000, (n,)):
        for m in torch.randint(1, 10000, (10,)):
            ones = torch.ones(m)
            index = torch.sort(torch.randint(0, target_size, (m,)))[0]

            a = segment_coo(ones, index, dim_size=target_size)
            b = torch.zeros(target_size).scatter_reduce(
                dim=0, index=index, src=ones, reduce="sum"
            )

            assert (a == b).all()


def test_segment_csr():
    src = torch.randn(100)
    indptr = torch.sort(torch.randperm(100)[:5])[0]

    assert sum_partitions(src, indptr).isclose(segment_csr(src, indptr)).all()
