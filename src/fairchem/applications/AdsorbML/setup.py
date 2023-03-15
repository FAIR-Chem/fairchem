"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

setup(
    name="adsorbml",
    version="0.0.1",
    description="Module for calculating the minima adsorbtion energy",
    url="http://github.com/Open-Catalyst-Project/AdsorbML",
    packages=find_packages(),
    include_package_data=True,
)
