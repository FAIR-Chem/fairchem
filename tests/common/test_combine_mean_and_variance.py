import pytest
import numpy as np

from ocpmodels.common.utils import combine_means_and_variances


def test_combine_mean_and_variance():
    rng = np.random.default_rng(12345)
    # generate 50000 random'ish samples with non trivial means / variances
    n = 50000
    values = (
        np.sin(rng.uniform(0, 1, n) + np.arange(n) * 0.1)
        + np.arange(n) * 0.005
    )

    expected_mean = values.mean()
    for ddof in [0, 1]:
        expected_variance = values.var(ddof=ddof)
        for _ in range(100):
            # chose 9 random partitions of the data
            vs = []
            idxs = sorted(np.floor(rng.uniform(0, 1, 9) * n).astype(int))
            for start_idx, end_idx in zip([0] + idxs, idxs + [n]):
                vs.append(values[start_idx:end_idx])
            # combine means and variances from random partitions
            calculated_mean, calculated_variance = combine_means_and_variances(
                [len(x) for x in vs],
                [x.var(ddof=ddof) for x in vs],
                [x.mean() for x in vs],
                ddof=ddof,
            )
            assert np.isclose(expected_mean, calculated_mean)
            assert np.isclose(expected_variance, calculated_variance)
