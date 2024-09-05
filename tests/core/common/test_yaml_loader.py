from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from fairchem.core.common.utils import UniqueKeyLoader, load_config


@pytest.fixture(scope="class")
def invalid_yaml_config():
    return """
key1:
    - a
    - b
key1:
    - c
    - d
"""


@pytest.fixture(scope="class")
def valid_yaml_config():
    return """
key1:
    - a
    - b
key2:
    - c
    - d
"""


@pytest.fixture(scope="class")
def include_path_in_yaml_config():
    return """
includes:
    - other.yml
"""


def test_invalid_config(invalid_yaml_config):
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(invalid_yaml_config.encode())
        fname = fp.name

    with pytest.raises(ValueError):  # noqa
        with open(fname) as fp:
            yaml.load(fp, Loader=UniqueKeyLoader)


def test_valid_config(valid_yaml_config):
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(valid_yaml_config.encode())
        fname = fp.name

    with open(fname) as fp:
        yaml.load(fp, Loader=UniqueKeyLoader)


def test_load_config_with_include_path(include_path_in_yaml_config, valid_yaml_config):
    with tempfile.TemporaryDirectory() as tempdirname:

        this_yml_path = f"{tempdirname}/this.yml"
        with open(this_yml_path, "w") as fp:
            fp.write(include_path_in_yaml_config)

        # the include does not exist throw an error!
        with pytest.raises(ValueError):
            load_config(this_yml_path)

        other_yml_path = f"{tempdirname}/subfolder"
        os.mkdir(other_yml_path)
        other_yml_full_filename = f"{other_yml_path}/other.yml"
        with open(other_yml_full_filename, "w") as fp:
            fp.write(valid_yaml_config)

        # the include does not exist throw an error!
        with pytest.raises(ValueError):
            load_config(this_yml_path)

        # the include does not exist throw an error!
        loaded_config = load_config(this_yml_path, include_paths=[other_yml_path])
        assert set(loaded_config[0].keys()) == set(["key1", "key2"])
