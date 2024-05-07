from __future__ import annotations

import torch

from fairchem.core.models.escn.so3 import CoefficientMapping
from fairchem.core.models.escn.so3 import SO3_Embedding as escn_SO3_Embedding


class TestMPrimaryLPrimary:
    def test_mprimary_lprimary_mappings(self):
        def sign(x):
            return 1 if x >= 0 else -1

        device = torch.device("cpu")
        lmax_list = [6, 8]
        mmax_list = [3, 6]
        for lmax in lmax_list:
            for mmax in mmax_list:
                c = CoefficientMapping([lmax], [mmax], device=device)

                escn_embedding = escn_SO3_Embedding(
                    length=1,
                    lmax_list=[lmax],
                    num_channels=1,
                    device=device,
                    dtype=torch.float32,
                )

                """
                Generate L_primary matrix
                L0: 0.00 ~ L0M0
                L1: -1.01 1.00 1.01 ~ L1M(-1),L1M0,L1M1
                L2: -2.02 -2.01 2.00 2.01 2.02 ~ L2M(-2),L2M(-1),L2M0,L2M1,L2M2
                """
                test_matrix_lp = []
                for l in range(lmax + 1):
                    max_m = min(l, mmax)
                    for m in range(-max_m, max_m + 1):
                        v = l * sign(m) + 0.01 * m  # +/- l . 00 m
                        test_matrix_lp.append(v)

                test_matrix_lp = (
                    torch.tensor(test_matrix_lp).reshape(1, -1, 1).to(torch.float32)
                )

                """
                Generate M_primary matrix
                M0: 0.00 , 1.00, 2.00, ... , LMax ~ M0L0, M0L1, .., M0L(LMax)
                M1: 1.01, 2.01, .., LMax.01, -1.01, -2.01, -LMax.01 ~ L1M1, L2M1, .., L(LMax)M1, L1M(-1), L2M(-1), ... , L(LMax)M(-1)
                """
                test_matrix_mp = []
                for m in range(max_m + 1):
                    for l in range(m, lmax + 1):
                        v = l + 0.01 * m  # +/- l . 00 m
                        test_matrix_mp.append(v)
                    if m > 0:
                        for l in range(m, lmax + 1):
                            v = -(l + 0.01 * m)  # +/- l . 00 m
                            test_matrix_mp.append(v)

                test_matrix_mp = (
                    torch.tensor(test_matrix_mp).reshape(1, -1, 1).to(torch.float32)
                )

                escn_embedding.embedding = test_matrix_lp.clone()

                escn_embedding._m_primary(c)
                mp = escn_embedding.embedding.clone()
                (test_matrix_mp == mp).all()

                escn_embedding._l_primary(c)
                lp = escn_embedding.embedding.clone()
                (test_matrix_lp == lp).all()
