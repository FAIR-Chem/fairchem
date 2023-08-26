"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from collections import defaultdict
from functools import cached_property
from typing import Literal, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from typing_extensions import Annotated, override

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.typed_config import Field, TypedConfig
from ocpmodels.common.utils import check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.base_trainer import BaseTrainer

from .optimizer import OptimConfig, load_optimizer


@registry.register_trainer("ft")
class FTTrainer(BaseTrainer):
    @cached_property
    def optim_config(self):
        return OptimConfig.from_dict(self.config["optimizer"])

    @override
    def load_optimizer(self) -> None:
        self.optimizer, self.lr_scheduler, self.ema = load_optimizer(
            self.model, self.optim_config
        )

    @override
    def load_extras(self) -> None:
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm", False)
