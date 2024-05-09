from __future__ import annotations

import tempfile

import pytest
import yaml

from fairchem.core.common.utils import UniqueKeyLoader


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
