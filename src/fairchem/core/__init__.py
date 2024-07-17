"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fairchem.core.common.relaxation.ase_utils import OCPCalculator

__all__ = ["OCPCalculator"]

try:
    __version__ = version("fairchem.core")
except PackageNotFoundError:
    # package is not installed
    __version__ = ""
