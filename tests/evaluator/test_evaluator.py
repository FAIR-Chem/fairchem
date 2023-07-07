"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import pytest
import torch

from ocpmodels.modules.evaluator import (
    Evaluator,
    cosine_similarity,
    magnitude_error,
)


@pytest.fixture(scope="class")
def load_evaluator_s2ef(request) -> None:
    request.cls.evaluator = Evaluator(task="s2ef")
    prediction = {
        "energy": torch.randn(6),
        "forces": torch.randn(1000000, 3),
        "natoms": torch.tensor(
            (100000, 200000, 300000, 200000, 100000, 100000)
        ),
    }
    target = {
        "energy": torch.randn(6),
        "forces": torch.randn(1000000, 3),
        "natoms": torch.tensor(
            (100000, 200000, 300000, 200000, 100000, 100000)
        ),
    }
    request.cls.metrics = request.cls.evaluator.eval(prediction, target)


@pytest.fixture(scope="class")
def load_evaluator_is2rs(request) -> None:
    request.cls.evaluator = Evaluator(task="is2rs")
    prediction = {
        "positions": torch.randn(50, 3),
        "natoms": torch.tensor((5, 5, 10, 12, 18)),
        "cell": torch.randn(5, 3, 3),
        "pbc": torch.tensor([True, True, True]),
    }
    target = {
        "positions": torch.randn(50, 3),
        "cell": torch.randn(5, 3, 3),
        "natoms": torch.tensor((5, 5, 10, 12, 18)),
        "pbc": torch.tensor([True, True, True]),
    }
    request.cls.metrics = request.cls.evaluator.eval(prediction, target)


@pytest.fixture(scope="class")
def load_evaluator_is2re(request) -> None:
    request.cls.evaluator = Evaluator(task="is2re")
    prediction = {
        "energy": torch.randn(50),
    }
    target = {
        "energy": torch.randn(50),
    }
    request.cls.metrics = request.cls.evaluator.eval(prediction, target)


class TestMetrics:
    def test_cosine_similarity(self) -> None:
        v1, v2 = torch.randn(1000000, 3), torch.randn(1000000, 3)
        res = cosine_similarity(v1, v2)
        np.testing.assert_almost_equal(res["metric"], 0, decimal=2)
        np.testing.assert_almost_equal(
            res["total"] / res["numel"], res["metric"]
        )

    def test_magnitude_error(self) -> None:
        v1, v2 = (
            torch.tensor([[0.0, 1], [-1, 0]]),
            torch.tensor([[0.0, 0], [0, 0]]),
        )
        res = magnitude_error(v1, v2)
        np.testing.assert_equal(res["metric"], 1.0)


@pytest.mark.usefixtures("load_evaluator_s2ef")
class TestS2EFEval:
    def test_metrics_exist(self) -> None:
        assert "energy_mae" in self.metrics
        assert "forces_mae" in self.metrics
        assert "forces_cos" in self.metrics
        assert "energy_force_within_threshold" in self.metrics


@pytest.mark.usefixtures("load_evaluator_is2rs")
class TestIS2RSEval:
    def test_metrics_exist(self) -> None:
        assert "average_distance_within_threshold" in self.metrics


@pytest.mark.usefixtures("load_evaluator_is2re")
class TestIS2REEval:
    def test_metrics_exist(self) -> None:
        assert "energy_mae" in self.metrics
        assert "energy_mse" in self.metrics
        assert "energy_within_threshold" in self.metrics
