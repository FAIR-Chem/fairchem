"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import setup

setup(
    name="ocpmodels",
    version="mila-0.1.0",
    description="Machine learning models for use in catalysis as part of the Open Catalyst Project",
    url="https://github.com/Open-Catalyst-Project/ocp",
    packages=["ocdata", "ocpmodels"],
    include_package_data=True,
)
