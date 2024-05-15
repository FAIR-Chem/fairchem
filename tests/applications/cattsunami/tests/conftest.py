from pathlib import Path
import pickle
import pytest


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
    with open(Path(__file__).parent / "autoframe_inputs_dissociation.pkl", "rb") as fp:
        request.cls.inputs = pickle.load(fp)


@pytest.fixture(scope="class")
def transfer_inputs(request):
    with open(Path(__file__).parent / "autoframe_inputs_transfer.pkl", "rb") as fp:
        request.cls.inputs = pickle.load(fp)
