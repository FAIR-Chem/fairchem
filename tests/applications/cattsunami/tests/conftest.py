import os
import pickle
from pathlib import Path

import pytest
from fairchem.core.scripts import download_large_files


@pytest.fixture(scope="class")
def neb_frames(request):
    with open(Path(__file__).parent / "neb_frames.pkl", "rb") as fp:
        request.cls.images = pickle.load(fp)


@pytest.fixture(scope="class")
def desorption_inputs(request):
    with open(Path(__file__).parent / "autoframe_inputs_desorption.pkl", "rb") as fp:
        request.cls.inputs = pickle.load(fp)


@pytest.fixture(scope="class")
def dissociation_inputs(request):
    pkl_path = Path(__file__).parent / "autoframe_inputs_dissociation.pkl"
    # We need to know where the test repo has been downloaded to,
    # it may be different from the src location, so we walk back up the
    # directory tree to see find where the `tests` directory is
    # This file's path is fairchem/tests/applications/cattsunami/tests
    # so parents[4] is `fairchem` which is what we want (it has the `tests`
    # directory)
    test_dir = Path(__file__).parents[4]
    if not pkl_path.exists():
        download_large_files.download_file_group("cattsunami", test_dir)
    with open(pkl_path, "rb") as fp:
        request.cls.inputs = pickle.load(fp)


@pytest.fixture(scope="class")
def transfer_inputs(request):
    pkl_path = Path(__file__).parent / "autoframe_inputs_transfer.pkl"
    # We need to know where the test repo has been downloaded to,
    # it may be different from the src location, so we walk back up the
    # directory tree to see find where the `tests` directory is
    # This file's path is fairchem/tests/applications/cattsunami/tests
    # so parents[4] is `fairchem` which is what we want (it has the `tests`
    # directory)
    test_dir = Path(__file__).parents[4]
    if not pkl_path.exists():
        download_large_files.download_file_group("cattsunami", test_dir)
    with open(pkl_path, "rb") as fp:
        request.cls.inputs = pickle.load(fp)
