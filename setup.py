"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path

from setuptools import setup

setup(
    name="ocpmodels",
    version="0.1.0",
    description="Machine learning models for use in catalysis as part of the Open Catalyst Project",
    url="https://github.com/Open-Catalyst-Project/ocp",
    packages=["ocdata"]
    + [str(p) for p in Path("./ocpmodels/").glob("**") if (p / "__init__.py").exists()],
    include_package_data=True,
)
