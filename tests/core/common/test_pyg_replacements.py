pyg_available = False
try:
    from torch_scatter import segment_coo, segment_csr

    pyg_available = True
except ImportError:
    pass

import torch
import pytest
from torch_scatter import scatter as  torch_scatter_scatter
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

def test_torch_scatter_scatter_vs_torch_scatter():
    n=128
    dim=0
    for k in [n//32,n,10*n,100*n]:
        index=torch.randint(0,n,(k,))
        src=torch.rand(k)

        out_a=torch.zeros(n).scatter_reduce_(dim,index,src,reduce='sum')

        out_b=torch.zeros(n)
        torch_scatter_scatter(src,index,dim,out_b)

        assert out_a.isclose(out_b).all()