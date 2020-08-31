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
        # "s2ef": [forcex_mae, forcex_mse, forcey_mae, forcey_mse, forcez_mae, forcez_mse, force_mae, force_mse, force_cos, energy_mae, energy_mse],
        # "is2rs": [positions_mae, positions_mse],
        "is2re": ["energy_mae", "energy_mse"],
    }

    task_attributes = {
        "s2ef": ["energy", "forces"],
        "is2rs": ["positions"],
        "is2re": ["energy"],
    }

    task_primary_metric = {
        "s2ef": "force_mae",
        "is2rs": "positions_mae",
        "is2re": "energy_mae",
    }

    def __init__(self, task=None):
        assert task in ["s2ef", "is2re", "is2re"]
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
    error = absolute_error(prediction["energy"], target["energy"])
    return {
        "metric": torch.mean(error, dim=0).item(),
        "total": torch.sum(error).item(),
        "numel": prediction["energy"].numel(),
    }


def energy_mse(prediction, target):
    error = squared_error(prediction["energy"], target["energy"])
    return {
        "metric": torch.mean(error, dim=0).item(),
        "total": torch.sum(error).item(),
        "numel": prediction["energy"].numel(),
    }


def absolute_error(prediction, target):
    return torch.abs(target - prediction)


def squared_error(prediction, target):
    return (target - prediction) ** 2
