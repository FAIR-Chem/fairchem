import torch


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor, device):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim=0).to(device)
        self.std = torch.std(tensor, dim=0).to(device)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
