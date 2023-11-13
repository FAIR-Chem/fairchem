import torch
from time import time
from collections import defaultdict
import numpy as np

from ocpmodels.common.dist_utils import synchronize


class Timer:
    def __init__(self, name, store={}, gpu=False, ignore=False):
        self.times = store
        self.name = name
        self.gpu = gpu
        self.ignore = ignore

    def __enter__(self):
        if self.ignore:
            return self
        if self.gpu:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time()
        return self

    def __exit__(self, *args):
        if self.ignore:
            return
        if self.gpu:
            self.end.record()
            torch.cuda.synchronize()
            synchronize()
            self.duration = self.start.elapsed_time(self.end) / 1000
        else:
            self.end = time()
            self.duration = self.end - self.start
        self.times[self.name].append(self.duration)


class Times:
    def __init__(self, gpu=False, ignore=False):
        self.times = defaultdict(list)
        self.timers = {}
        self.gpu = gpu
        self.ignore = ignore

    def reset(self, keys=None):
        """
        Resets timers as per ``keys``.
        If ``keys`` is None, resets all timers.

        Args:
            keys (Union[str, List[str]], optional): Specific named timers to reset,
            or all of them if ``keys`` is ``None`` . Defaults to ``None``.
        """
        if isinstance(keys, str):
            keys = [keys]

        if keys is None:
            self.times = defaultdict(list)
            self.timers = {}
        else:
            for k in keys:
                if k in self.times:
                    self.times[k] = []
                self.timers.pop(k, None)

    def prepare_for_logging(self, map_func=None, map_funcs=None):
        """
        Computes mean and standard deviation of all timers.
        Returns a tuple: (mean_times_dict, std_times_dict)

        Returns:
            tuple[dict]: a dict with mean times and a dict with std times
        """
        mean_times = {}
        std_times = {}
        for k, v in self.times.items():
            if map_funcs is not None:
                data = list(map(map_funcs.get(k, lambda x: x), v))
            elif map_func is not None:
                data = list(map(map_func, v))
            else:
                data = v
            mean_times[k] = np.mean(data)
            std_times[k] = np.std(data)
        return mean_times, std_times

    def next(self, name, ignore=None, gpu=None):
        if "name" not in self.timers:
            if ignore is None:
                ignore = self.ignore
            self.timers[name] = Timer(
                name, self.times, self.gpu if gpu is None else gpu, ignore
            )
        if ignore != self.timers[name].ignore:
            self.timers[name].ignore = ignore
        return self.timers[name]
