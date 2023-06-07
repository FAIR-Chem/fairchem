import torch

from ..config import TypedConfig


class NormalizerConfig(TypedConfig):
    enabled: bool = True

    mean: float = 0.0
    std: float = 1.0


class Normalizer:
    def __init__(self, config: NormalizerConfig):
        self.config = config

    def normalize(self, x: torch.Tensor):
        if not self.config.enabled:
            return x
        return (x - self.config.mean) / self.config.std

    def denormalize(self, x: torch.Tensor):
        if not self.config.enabled:
            return x
        return x * self.config.std + self.config.mean
