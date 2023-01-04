import torch
from time import time, sleep
from collections import defaultdict
import numpy as np


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

    def reset(self):
        self.times = defaultdict(list)
        self.timers = {}

    def prepare_for_logging(self):
        """
        Computes mean and standard deviation of all timers.
        Returns a tuple: (mean_times_dict, std_times_dict)

        Returns:
            tuple[dict]: a dict with mean times and a dict with std times
        """
        mean_times = {}
        std_times = {}
        for k, v in self.times.items():
            mean_times[k] = np.mean(v)
            std_times[k] = np.std(v)
        return mean_times, std_times

    def next(self, name, ignore=None):
        if "name" not in self.timers:
            if ignore is None:
                ignore = self.ignore
            self.timers[name] = Timer(name, self.times, self.gpu, ignore)
        return self.timers[name]


if __name__ == "__main__":

    times = Times(gpu=True)
    with times.next("a"):
        sleep(0.1)
    with times.next("b"):
        sleep(0.2)
    with times.next("a"):
        sleep(0.3)
    with times.next("b"):
        sleep(0.4)
    with times.next("a"):
        sleep(0.5)
    with times.next("b"):
        sleep(0.6)
    print(times.prepare_for_logging())
