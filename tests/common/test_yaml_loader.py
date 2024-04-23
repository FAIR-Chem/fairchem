import tempfile

import pytest
import yaml

from ocpmodels.common.utils import UniqueKeyLoader


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
        fp.close()
        with pytest.raises(ValueError):
            yaml.load(open(fp.name, "r"), Loader=UniqueKeyLoader)


def test_valid_config(valid_yaml_config):
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(valid_yaml_config.encode())
        fp.close()
        yaml.load(open(fp.name, "r"), Loader=UniqueKeyLoader)
