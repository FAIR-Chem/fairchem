"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

setup(
    name="omdata",
    version="0.1.0",
    description="Code for generating OMOL input configurations",
    url="http://github.com/Open-Catalyst-Project/om-data",
    packages=find_packages(),
    install_requires=[
        "ase@git+https://gitlab.com/ase/ase.git@dc86a19a280741aa2b42a08d0fa63a8d0348e225",
        "quacc[sella]>=0.7.6",
        "sella==2.3.3",
    ],
    include_package_data=True,
)
