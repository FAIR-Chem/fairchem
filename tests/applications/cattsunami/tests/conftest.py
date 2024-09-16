from __future__ import annotations

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
    if not pkl_path.exists():
        download_large_files.download_file_group("cattsunami")
    with open(pkl_path, "rb") as fp:
        request.cls.inputs = pickle.load(fp)


@pytest.fixture(scope="class")
def transfer_inputs(request):
    pkl_path = Path(__file__).parent / "autoframe_inputs_transfer.pkl"
    if not pkl_path.exists():
        download_large_files.download_file_group("cattsunami")
    with open(pkl_path, "rb") as fp:
        request.cls.inputs = pickle.load(fp)
