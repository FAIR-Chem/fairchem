"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

setup(
    name="ocdata",
    version="0.2.0",
    description="Code for generating adsorbate-catalyst input configurations",
    url="http://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset",
    packages=find_packages(),
    package_data={"ocdata.databases.pkls": ["*pkl"]},
    include_package_data=True,
)
