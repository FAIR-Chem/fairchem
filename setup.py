"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from distutils.util import convert_path
from pathlib import Path

from setuptools import setup


def make_ocpmodels_package_dict():
    dirs = [
        convert_path(str(p))
        for p in Path("./ocpmodels/").glob("**")
        if (p / "__init__.py").exists()
    ]
    pkgs = [d.replace("/", ".") for d in dirs]
    return {p: d for p, d in zip(pkgs, dirs)}


pkg_dict = make_ocpmodels_package_dict()
pkg_dict["ocdata"] = convert_path("ocdata")

setup(
    name="ocpmodels",
    version="0.1.0",
    description="Machine learning models for use in catalysis as part of the Open Catalyst Project",
    url="https://github.com/Open-Catalyst-Project/ocp",
    packages=list(pkg_dict.keys()),
    package_dir=pkg_dict,
    include_package_data=True,
)
