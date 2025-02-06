from __future__ import annotations

from fairchem.core.common.utils import get_deep


def test_get_deep() -> None:
    d = {"oc20": {"energy": 1.5}}
    assert get_deep(d, "oc20.energy") == 1.5
    assert get_deep(d, "oc20.force", 0.9) == 0.9
    assert get_deep(d, "omol.energy") is None
