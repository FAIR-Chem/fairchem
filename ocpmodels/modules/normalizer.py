"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None):
        """tensor is taken as a sample to calculate the mean and std"""
        if tensor is None and mean is None:
            return

        if device is None:
            device = "cpu"

        self.device = device

        if tensor is not None:
            self.mean = torch.mean(tensor, dim=0).to(device)
            self.std = torch.std(tensor, dim=0).to(device)
            return

        if mean is not None and std is not None:
            self.mean = torch.tensor(mean).to(device)
            self.std = torch.tensor(std).to(device)

        self.hof_mean = None
        self.hof_std = None
        self.rescale_with_hof = False

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if self.hof_mean:
            self.hof_mean = self.hof_mean.to(device)
        if self.hof_std:
            self.hof_std = self.hof_std.to(device)
        self.device = device

    def norm(self, tensor, hofs=None):
        if hofs is not None and self.rescale_with_hof:
            return tensor / hofs - self.hof_mean
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor, hofs=None):
        if hofs is not None and self.rescale_with_hof:
            return (normed_tensor + self.hof_mean) * hofs
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        sd = {"mean": self.mean, "std": self.std}
        if self.rescale_with_hof:
            sd["hof_stats"] = {
                "mean": self.hof_mean,
                "std": self.hof_std,
            }
        return sd

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].to(self.mean.device)
        self.std = state_dict["std"].to(self.mean.device)
        if "hof_stats" in state_dict:
            self.set_hof_rescales(state_dict["hof_stats"])

    def set_hof_rescales(self, hof_stats):
        self.hof_mean = torch.tensor(hof_stats["mean"], device=self.device)
        self.hof_std = torch.tensor(hof_stats["std"], device=self.device)
        self.rescale_with_hof = True
