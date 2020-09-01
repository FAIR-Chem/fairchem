# THIS IS NOW DEPRECATED. Consider using ocpmodels.modules.evaluator instead.
from collections import defaultdict, deque

import torch


# from github.com/facebookresearch/pythia/blob/12f67cd4f67499814bb0b3665ff14dd635800f63/pythia/common/meter.py
class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.deque = deque(maxlen=self.window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    def get_latest(self):
        return self.deque[-1]


class Meter:
    def __init__(self, delimiter=", ", split="train"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.split = split

    def update(self, update_dict):
        for k, v in update_dict.items():
            if isinstance(v, torch.Tensor):
                if v.dim() != 0:
                    v = v.mean()
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_from_meter(self, meter):
        for key, value in meter.meters.items():
            assert isinstance(value, SmoothedValue)
            self.meters[key] = value

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, attr
            )
        )

    def get_scalar_dict(self):
        scalar_dict = {}
        for k, v in self.meters.items():
            scalar_dict[k] = v.global_avg

        return scalar_dict

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # Regardless of split, if "epoch" or "step", print latest.
            if "epoch" in name or "step" in name:
                loss_str.append("{}: {:.4f}".format(name, meter.get_latest()))
            # If training split, print mean over the past window_size points.
            elif "train" in self.split:
                loss_str.append("{}: {:.4f}".format(name, meter.avg))
            # If val / test splits, print global average over the entire split.
            elif "val" in self.split or "test" in self.split:
                loss_str.append("{}: {:.4f}".format(name, meter.global_avg))
            else:
                raise NotImplementedError

        return self.delimiter.join(loss_str)


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction), dim=0)


def mae_ratio(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    divided by the absolute values of target

    Parameters
    ----------

    prediction: torch.Tensor (N, T)
    target: torch.Tensor (N, T)
    """
    return torch.mean(
        torch.abs(target - prediction) / (torch.abs(target) + 1e-7), dim=0
    )


def mean_l2_distance(prediction, target):
    """
    Computes the mean atomic distances

    Parameters
    ----------

    prediction: torch.Tensor (N, 3)
    target: torch.Tensor (N, 3)

    Return
    ----------
    avg distance: (N,1)
    """
    dist = torch.sqrt(torch.sum((target - prediction) ** 2, dim=1))
    return torch.mean(dist)
