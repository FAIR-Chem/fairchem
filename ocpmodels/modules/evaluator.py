"""
An evaluation module for use with the OCP dataset and suite of tasks. It should
be possible to import this independently of the rest of the codebase, e.g:

```
from ocpmodels.modules import Evaluator

evaluator = Evaluator(task="is2re")
perf = evaluator.eval(prediction, target)
```

task: "s2ef", "is2rs", "is2re".

We specify a default set of metrics for each task, but should be easy to extend
to add more metrics. `evaluator.eval` takes as input two dictionaries, one for
predictions and another for targets to check against. It returns a dictionary
with the relevant metrics computed.
"""

import torch


class Evaluator:
    task_metrics = {
        "s2ef": [
            "forcesx_mae",
            "forcesy_mae",
            "forcesz_mae",
            "forces_mae",
            "forces_cos",
            "forces_magnitude",
            "energy_mae",
        ],
        "is2rs": ["positions_mae", "positions_mse"],
        "is2re": ["energy_mae", "energy_mse"],
    }

    task_attributes = {
        "s2ef": ["energy", "forces"],
        "is2rs": ["positions"],
        "is2re": ["energy"],
    }

    task_primary_metric = {
        "s2ef": "forces_mae",
        "is2rs": "positions_mae",
        "is2re": "energy_mae",
    }

    def __init__(self, task=None):
        assert task in ["s2ef", "is2rs", "is2re"]
        self.task = task
        self.metric_fn = self.task_metrics[task]

    def eval(self, prediction, target, prev_metrics={}):
        for attr in self.task_attributes[self.task]:
            assert attr in prediction
            assert attr in target
            assert prediction[attr].shape == target[attr].shape

        metrics = prev_metrics

        for fn in self.task_metrics[self.task]:
            res = eval(fn)(prediction, target)
            metrics = self.update(fn, res, metrics)

        return metrics

    def update(self, key, stat, metrics):
        if key not in metrics:
            metrics[key] = {
                "metric": None,
                "total": 0,
                "numel": 0,
            }

        if isinstance(stat, dict):
            # If dictionary, we expect it to have `metric`, `total`, `numel`.
            metrics[key]["total"] += stat["total"]
            metrics[key]["numel"] += stat["numel"]
            metrics[key]["metric"] = (
                metrics[key]["total"] / metrics[key]["numel"]
            )
        elif isinstance(stat, float) or isinstance(stat, int):
            # If float or int, just add to the total and increment numel by 1.
            metrics[key]["total"] += stat
            metrics[key]["numel"] += 1
            metrics[key]["metric"] = (
                metrics[key]["total"] / metrics[key]["numel"]
            )
        elif torch.is_tensor(stat):
            raise NotImplementedError

        return metrics


def energy_mae(prediction, target):
    return absolute_error(prediction["energy"], target["energy"])


def energy_mse(prediction, target):
    return squared_error(prediction["energy"], target["energy"])


def forcesx_mae(prediction, target):
    return absolute_error(prediction["forces"][:, 0], target["forces"][:, 0])


def forcesx_mse(prediction, target):
    return squared_error(prediction["forces"][:, 0], target["forces"][:, 0])


def forcesy_mae(prediction, target):
    return absolute_error(prediction["forces"][:, 1], target["forces"][:, 1])


def forcesy_mse(prediction, target):
    return squared_error(prediction["forces"][:, 1], target["forces"][:, 1])


def forcesz_mae(prediction, target):
    return absolute_error(prediction["forces"][:, 2], target["forces"][:, 2])


def forcesz_mse(prediction, target):
    return squared_error(prediction["forces"][:, 2], target["forces"][:, 2])


def forces_mae(prediction, target):
    return absolute_error(prediction["forces"], target["forces"])


def forces_mse(prediction, target):
    return squared_error(prediction["forces"], target["forces"])


def forces_cos(prediction, target):
    return cosine_similarity(prediction["forces"], target["forces"])


def forces_magnitude(prediction, target):
    return magnitude_error(prediction["forces"], target["forces"], p=2)


def positions_mae(prediction, target):
    return absolute_error(prediction["positions"], target["positions"])


def positions_mse(prediction, target):
    return squared_error(prediction["positions"], target["positions"])


def cosine_similarity(prediction, target):
    error = torch.cosine_similarity(prediction, target)
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": error.numel(),
    }


def absolute_error(prediction, target):
    error = torch.abs(target - prediction)
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": prediction.numel(),
    }


def squared_error(prediction, target):
    error = (target - prediction) ** 2
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": prediction.numel(),
    }


def magnitude_error(prediction, target, p=2):
    assert prediction.shape[1] > 1
    error = torch.abs(
        torch.norm(prediction, p=p, dim=-1) - torch.norm(target, p=p, dim=-1)
    )
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": error.numel(),
    }
