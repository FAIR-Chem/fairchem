import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


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
        torch.abs(target - prediction) / (torch.abs(target) + 1e-7)
    )
