# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "BaseModel",
    "CGCNN",
    "DimeNet",
    "DimeNetPlusPlus",
    "SchNet",
    "ForceNet",
    "SphereNet",
]

from .base import BaseModel
from .cgcnn import CGCNN
from .dimenet import DimeNetWrap as DimeNet
from .dimenet_plus_plus import DimeNetPlusPlusWrap as DimeNetPlusPlus
from .forcenet import ForceNet
from .schnet import SchNetWrap as SchNet
from .spherenet import SphereNet

DimeNet.__module__ = __name__
DimeNet.__name__ = "DimeNet"

DimeNetPlusPlus.__module__ = __name__
DimeNetPlusPlus.__name__ = "DimeNetPlusPlus"

SchNet.__module__ = __name__
SchNet.__name__ = "SchNet"

ForceNet.__module__ = __name__
ForceNet.__name__ = "ForceNet"

SphereNet.__module__ = __name__
SphereNet.__name__ = "SphereNet"
