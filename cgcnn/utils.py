import os
import shutil

import numpy as np
import torch
from sklearn import metrics


def save_checkpoint(state, is_best, checkpoint_dir="checkpoints/"):
    filename = os.path.join(checkpoint_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, os.path.join(checkpoint_dir, "model_best.pth.tar")
        )
