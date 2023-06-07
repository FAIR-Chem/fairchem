# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
from weakref import ref

import pytest

from .local.debug import DebugExecutor
from .local.local import LocalExecutor


def job_with_weakref(ex):
    class MyObject:
        hello = "world"

    a = MyObject()
    a_ref = ref(a)
    assert a_ref() is a

    def f(a_ref):
        a = a_ref()
        assert a is not None
        return a_ref().hello

    return ex.submit(f, ref(a))


@pytest.mark.xfail(reason="'a' is GC-ed before we call the function")
def test_weakref_no_pickle(tmp_path):
    ex = DebugExecutor(tmp_path)
    assert job_with_weakref(ex).result() == "world"


@pytest.mark.xfail(reason="'ref(a)' can't be pickled")
def test_weakref_with_pickle(tmp_path):
    ex = LocalExecutor(tmp_path)
    assert job_with_weakref(ex).result() == "world"


def hello_fn() -> None:
    print("hello world")


def test_nested_pickling(tmp_path):
    def make_pickle() -> bytes:
        return pickle.dumps(hello_fn)

    pkl = make_pickle()
    assert bytes(__name__, "ascii") in pkl
    assert b"hello_fn" in pkl
    ex = LocalExecutor(tmp_path)
    j = ex.submit(make_pickle)
    assert j.result() == pkl


@pytest.mark.xfail(reason="Submitit changes __main__")
def test_submitit_respects_main(tmp_path):
    # TODO: I think this is the root cause of issue #11
    # https://github.com/facebookincubator/submitit/issues/11
    # Some programs like pytorch-lightning are dependent on the value of __main__
    # See how `pdb` manage to restore the correct __main__:
    # https://sourcegraph.com/github.com/python/cpython/-/blob/Lib/pdb.py#L1549
    # But maybe we could fix #11 by just using
    # `from submitit.core.submission import submitit_main`
    # as in https://github.com/facebookincubator/submitit/issues/11#issuecomment-713148952

    def get_main() -> str:
        # pylint: disable=import-outside-toplevel
        import __main__  # type: ignore

        return getattr(__main__, "__file__", "")

    main = get_main()
    ex = LocalExecutor(tmp_path)
    j_main = ex.submit(get_main).result()
    assert main == j_main
