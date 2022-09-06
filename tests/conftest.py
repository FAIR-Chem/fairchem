from typing import TYPE_CHECKING, Union

import numpy as np
import pytest
from syrupy.extensions.amber import AmberSnapshotExtension


if TYPE_CHECKING:
    from syrupy.types import SerializableData

DEFAULT_RTOL = 1.0e-05
DEFAULT_ATOL = 1.0e-08


class Approx:
    def __init__(
        self,
        data: Union[np.ndarray, list],
        *,
        rtol: float,
        atol: float,
    ):
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError(f"Cannot convert {type(data)} to np.array")

        self.rtol = rtol
        self.atol = atol

    def __repr__(self):
        data = np.array_repr(self.data)
        data = "\n".join(f"\t{line}" for line in data.splitlines())
        return (
            f"Approx(\n{data}, \n\trtol={self.rtol}, \n\tatol={self.atol}\n)"
        )


class _ApproxNumpyFormatter:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        rtol = self.data.rel or DEFAULT_RTOL
        atol = self.data.abs or DEFAULT_ATOL
        return Approx(self.data.expected, rtol=rtol, atol=atol).__repr__()


def _approx_from_string(data_str: str):
    """
    Parse the string representation of an Approx object.
    We can just use eval here, since we know the string is safe.
    """

    return eval(
        data_str.replace("dtype=", "dtype=np."),
        {"Approx": Approx, "np": np},
        {"array": np.array},
    )


class ApproxExtension(AmberSnapshotExtension):
    def matches(
        self,
        *,
        serialized_data: "SerializableData",
        snapshot_data: "SerializableData",
    ) -> bool:
        if isinstance(serialized_data, str):
            serialized_data_ = serialized_data.strip()
            if serialized_data_.startswith("Approx("):
                serialized_data = _approx_from_string(serialized_data_)
        if isinstance(snapshot_data, str):
            snapshot_data_ = snapshot_data.strip()
            if snapshot_data_.startswith("Approx("):
                snapshot_data = _approx_from_string(snapshot_data_)

        if not isinstance(serialized_data, Approx) or not isinstance(
            snapshot_data, Approx
        ):
            return super().matches(
                serialized_data=serialized_data, snapshot_data=snapshot_data
            )

        return np.allclose(
            snapshot_data.data,
            serialized_data.data,
            rtol=serialized_data.rtol,
            atol=serialized_data.atol,
        )

    def serialize(self, data, **kwargs):
        if isinstance(data, type(pytest.approx(np.array(0.0)))):
            return super().serialize(_ApproxNumpyFormatter(data), **kwargs)
        elif isinstance(data, type(pytest.approx(0.0))):
            raise NotImplementedError("Scalar approx not implemented yet")
        return super().serialize(data, **kwargs)


@pytest.fixture
def snapshot(snapshot):
    return snapshot.use_extension(ApproxExtension)
