import numpy as np
import pytest
import torch

from ocpmodels.modules.evaluator import Evaluator


@pytest.fixture(scope="class")
def load_evaluator_s2ef(request):
    request.cls.evaluator = Evaluator(task="s2ef")


@pytest.fixture(scope="class")
def load_evaluator_is2rs(request):
    request.cls.evaluator = Evaluator(task="is2rs")


@pytest.fixture(scope="class")
def load_evaluator_is2re(request):
    request.cls.evaluator = Evaluator(task="is2re")


@pytest.mark.usefixtures("load_evaluator_s2ef")
class TestS2EFEval:
    def test_s2ef_metrics(self):
        prediction = {
            "energy": torch.randn(50),
            "forces": torch.randn(1000000, 3),
        }
        target = {
            "energy": torch.randn(50),
            "forces": torch.randn(1000000, 3),
        }

        metrics = self.evaluator.eval(prediction, target)

        assert "energy_mae" in metrics
        assert "forces_mae" in metrics
        assert "forces_cos" in metrics

        np.testing.assert_almost_equal(
            metrics["forces_cos"]["metric"], 0, decimal=4
        )


@pytest.mark.usefixtures("load_evaluator_is2rs")
class TestIS2RSEval:
    def test_is2rs_metrics(self):
        prediction = {
            "positions": torch.randn(50, 3),
        }
        target = {
            "positions": torch.randn(50, 3),
        }

        metrics = self.evaluator.eval(prediction, target)

        assert "positions_mae" in metrics
        assert "positions_mse" in metrics


@pytest.mark.usefixtures("load_evaluator_is2re")
class TestIS2REEval:
    def test_is2re_metrics(self):
        prediction = {
            "energy": torch.randn(50),
        }
        target = {
            "energy": torch.randn(50),
        }

        metrics = self.evaluator.eval(prediction, target)

        assert "energy_mae" in metrics
        assert "energy_mse" in metrics
