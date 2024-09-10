from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def max_num_elements(dummy_element_refs):
    return len(dummy_element_refs) - 1
