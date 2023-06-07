# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from pathlib import Path

import pytest

from .local.local import LocalExecutor


@pytest.fixture()
def executor(tmp_path: Path) -> LocalExecutor:
    return LocalExecutor(tmp_path)


@pytest.fixture(params=["a_0", "a 0", 'a"=0"', "a'; echo foo", r"a\=0", r"a\=", "a\n0"])
def weird_tmp_path(request, tmp_path: Path) -> Path:
    return tmp_path / request.param


@pytest.fixture()
def fast_forward_clock(monkeypatch):
    """Allows to go in the future."""
    clock_time = [time.time()]

    monkeypatch.setattr(time, "time", lambda: clock_time[0])

    def _fast_forward(minutes: float):
        clock_time[0] += minutes * 60

    return _fast_forward
