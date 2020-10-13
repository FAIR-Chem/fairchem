# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "BaseModel",
    "CGCNN",
    "DimeNet",
    "SchNet",
]

from .base import BaseModel
from .cgcnn import CGCNN
from .dimenet import DimeNetWrap as DimeNet
from .schnet import SchNetWrap as SchNet

DimeNet.__module__ = __name__
DimeNet.__name__ = "DimeNet"

SchNet.__module__ = __name__
SchNet.__name__ = "SchNet"
