from __future__ import annotations

import json
from dataclasses import dataclass
from typing import (
    Any,
    Final,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from unittest import TestCase as UnitTestCase

import numpy as np
from ase.atoms import Atoms as ASEAtoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from fairchem.demo.ocpapi.client import Status
from fairchem.demo.ocpapi.client.models import (
    Adsorbates,
    AdsorbateSlabConfigs,
    AdsorbateSlabRelaxationResult,
    AdsorbateSlabRelaxationsRequest,
    AdsorbateSlabRelaxationsResults,
    AdsorbateSlabRelaxationsSystem,
    Atoms,
    Bulk,
    Bulks,
    Model,
    Models,
    Slab,
    SlabMetadata,
    Slabs,
    _DataModel,
)

T = TypeVar("T", bound=_DataModel)


class ModelTestWrapper:
    class ModelTest(UnitTestCase, Generic[T]):
        """
        Base class for all tests below that assert behavior of data models.
        """

        def __init__(
            self,
            *args: Any,
            obj: T,
            obj_json: str,
            **kwargs: Any,
        ) -> None:
            """
            Args:
                obj: A model instance in which all fields, even unknown ones,
                    are included.
                obj_json: JSON-serialized version of obj.
            """
            super().__init__(*args, **kwargs)
            self._obj = obj
            self._obj_json = obj_json
            self._obj_type = type(obj)

        def test_from_json(self) -> None:
            @dataclass
            class TestCase:
                message: str
                json_repr: str
                expected: Final[Optional[T]] = None
                expected_exception: Final[Optional[Type[Exception]]] = None

            test_cases: List[TestCase] = [
                # If the json object is empty then default values should
                # be used for all fields
                TestCase(
                    message="empty object",
                    json_repr="{}",
                    expected_exception=Exception,
                ),
                # If all fields are set then they should be included in the
                # resulting object
                TestCase(
                    message="all fields set",
                    json_repr=self._obj_json,
                    expected=self._obj,
                ),
            ]

            for case in test_cases:
                with self.subTest(msg=case.message):
                    # Make sure an exception is raised if one is expected
                    if case.expected_exception is not None:
                        with self.assertRaises(case.expected_exception):
                            self._obj_type.from_json(case.json_repr)

                    # Otherwise make sure the expected value is returned
                    if case.expected is not None:
                        actual = self._obj_type.from_json(case.json_repr)
                        self.assertEqual(actual, case.expected)

        def test_to_json(self) -> None:
            @dataclass
            class TestCase:
                message: str
                obj: T
                expected: str

            test_cases: List[TestCase] = [
                # All explicitly-defined fields should serialize
                TestCase(
                    message="all fields set",
                    obj=self._obj,
                    expected=self._obj_json,
                ),
            ]

            for case in test_cases:
                with self.subTest(msg=case.message):
                    actual = case.obj.to_json()
                    self.assertJsonEqual(actual, case.expected)

        def assertJsonEqual(self, first: str, second: str) -> None:
            """
            Compares two JSON-formatted strings by deserializing them and then
            comparing the generated built-in types.
            """
            self.assertEqual(json.loads(first), json.loads(second))


class TestModel(ModelTestWrapper.ModelTest[Model]):
    """
    Serde tests for the Model data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=Model(
                id="model_id",
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "id": "model_id",
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestModels(ModelTestWrapper.ModelTest[Models]):
    """
    Serde tests for the Models data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=Models(
                models=[Model(id="model_id")],
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "models": [
        {
            "id": "model_id"
        }
    ],
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestBulk(ModelTestWrapper.ModelTest[Bulk]):
    """
    Serde tests for the Bulk data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=Bulk(
                src_id="test_id",
                elements=["A", "B"],
                formula="AB2",
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "src_id": "test_id",
    "els": ["A", "B"],
    "formula": "AB2",
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestBulks(ModelTestWrapper.ModelTest[Bulks]):
    """
    Serde tests for the Bulks data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=Bulks(
                bulks_supported=[
                    Bulk(
                        src_id="test_id",
                        elements=["A", "B"],
                        formula="AB2",
                    )
                ],
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "bulks_supported": [
        {
            "src_id": "test_id",
            "els": ["A", "B"],
            "formula": "AB2"
        }
    ],
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestAdsorbates(ModelTestWrapper.ModelTest[Adsorbates]):
    """
    Serde tests for the Adsorbates data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=Adsorbates(
                adsorbates_supported=["A", "B"],
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "adsorbates_supported": ["A", "B"],
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestAtoms(ModelTestWrapper.ModelTest[Atoms]):
    """
    Serde tests for the Atoms data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=Atoms(
                cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                pbc=(True, False, True),
                numbers=[1, 2],
                positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                tags=[0, 1],
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
    "pbc": [true, false, true],
    "numbers": [1, 2],
    "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
    "tags": [0, 1],
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )

    def test_to_ase_atoms(self) -> None:
        atoms = Atoms(
            cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
            pbc=(True, False, True),
            numbers=[1, 2],
            positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
            tags=[0, 1],
        )
        actual = atoms.to_ase_atoms()
        expected = ASEAtoms(
            cell=[(1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)],
            pbc=(True, False, True),
            numbers=[1, 2],
            positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
            tags=[0, 1],
            constraint=FixAtoms(mask=[True, False]),
        )
        self.assertEqual(actual, expected)
        # The constraint property isn't checked in the Atoms.__eq__ method
        # so check it explicitly
        self.assertEqual(actual.constraints[0].index, [0])


class TestSlabMetadata(ModelTestWrapper.ModelTest[SlabMetadata]):
    """
    Serde tests for the SlabMetadata data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=SlabMetadata(
                bulk_src_id="test_id",
                millers=(-1, 0, 1),
                shift=0.25,
                top=False,
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "bulk_id": "test_id",
    "millers": [-1, 0, 1],
    "shift": 0.25,
    "top": false,
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestSlab(ModelTestWrapper.ModelTest[Slab]):
    """
    Serde tests for the Slab data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=Slab(
                atoms=Atoms(
                    cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                    pbc=(True, False, True),
                    numbers=[1, 2],
                    positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                    tags=[0, 1],
                    other_fields={"extra_atoms_field": "extra_atoms_value"},
                ),
                metadata=SlabMetadata(
                    bulk_src_id="test_id",
                    millers=(-1, 0, 1),
                    shift=0.25,
                    top=False,
                    other_fields={"extra_metadata_field": "extra_metadata_value"},
                ),
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "slab_atomsobject": {
        "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
        "pbc": [true, false, true],
        "numbers": [1, 2],
        "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
        "tags": [0, 1],
        "extra_atoms_field": "extra_atoms_value"
    },
    "slab_metadata": {
        "bulk_id": "test_id",
        "millers": [-1, 0, 1],
        "shift": 0.25,
        "top": false,
        "extra_metadata_field": "extra_metadata_value"
    },
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestSlabs(ModelTestWrapper.ModelTest[Slabs]):
    """
    Serde tests for the Slabs data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=Slabs(
                slabs=[
                    Slab(
                        atoms=Atoms(
                            cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                            pbc=(True, False, True),
                            numbers=[1, 2],
                            positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                            tags=[0, 1],
                            other_fields={"extra_atoms_field": "extra_atoms_value"},
                        ),
                        metadata=SlabMetadata(
                            bulk_src_id="test_id",
                            millers=(-1, 0, 1),
                            shift=0.25,
                            top=False,
                            other_fields={
                                "extra_metadata_field": "extra_metadata_value"
                            },
                        ),
                        other_fields={"extra_slab_field": "extra_slab_value"},
                    )
                ],
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "slabs": [{
        "slab_atomsobject": {
            "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            "pbc": [true, false, true],
            "numbers": [1, 2],
            "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
            "tags": [0, 1],
            "extra_atoms_field": "extra_atoms_value"
        },
        "slab_metadata": {
            "bulk_id": "test_id",
            "millers": [-1, 0, 1],
            "shift": 0.25,
            "top": false,
            "extra_metadata_field": "extra_metadata_value"
        },
        "extra_slab_field": "extra_slab_value"
    }],
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestAdsorbateSlabConfigs(ModelTestWrapper.ModelTest[AdsorbateSlabConfigs]):
    """
    Serde tests for the AdsorbateSlabConfigs data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=AdsorbateSlabConfigs(
                adsorbate_configs=[
                    Atoms(
                        cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                        pbc=(True, False, True),
                        numbers=[1, 2],
                        positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                        tags=[0, 1],
                        other_fields={"extra_ad_atoms_field": "extra_ad_atoms_value"},
                    ),
                ],
                slab=Slab(
                    atoms=Atoms(
                        cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                        pbc=(True, False, True),
                        numbers=[1, 2],
                        positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                        tags=[0, 1],
                        other_fields={
                            "extra_slab_atoms_field": "extra_slab_atoms_value"
                        },
                    ),
                    metadata=SlabMetadata(
                        bulk_src_id="test_id",
                        millers=(-1, 0, 1),
                        shift=0.25,
                        top=False,
                        other_fields={"extra_metadata_field": "extra_metadata_value"},
                    ),
                    other_fields={"extra_slab_field": "extra_slab_value"},
                ),
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "adsorbate_configs": [
        {
            "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            "pbc": [true, false, true],
            "numbers": [1, 2],
            "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
            "tags": [0, 1],
            "extra_ad_atoms_field": "extra_ad_atoms_value"
        }
    ],
    "slab": {
        "slab_atomsobject": {
            "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            "pbc": [true, false, true],
            "numbers": [1, 2],
            "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
            "tags": [0, 1],
            "extra_slab_atoms_field": "extra_slab_atoms_value"
        },
        "slab_metadata": {
            "bulk_id": "test_id",
            "millers": [-1, 0, 1],
            "shift": 0.25,
            "top": false,
            "extra_metadata_field": "extra_metadata_value"
        },
        "extra_slab_field": "extra_slab_value"
    },
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestAdsorbateSlabRelaxationsSystem(
    ModelTestWrapper.ModelTest[AdsorbateSlabRelaxationsSystem]
):
    """
    Serde tests for the AdsorbateSlabRelaxationsSystem data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=AdsorbateSlabRelaxationsSystem(
                system_id="test_id",
                config_ids=[1, 2, 3],
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "system_id": "test_id",
    "config_ids": [1, 2, 3],
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestAdsorbateSlabRelaxationsRequest(
    ModelTestWrapper.ModelTest[AdsorbateSlabRelaxationsRequest]
):
    """
    Serde tests for the AdsorbateSlabRelaxationsRequest data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=AdsorbateSlabRelaxationsRequest(
                adsorbate="ABC",
                adsorbate_configs=[
                    Atoms(
                        cell=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)),
                        pbc=(True, False, True),
                        numbers=[1, 2],
                        positions=[(1.1, 1.2, 1.3), (1.4, 1.5, 1.6)],
                        tags=[2, 2],
                        other_fields={"extra_ad_field": "extra_ad_value"},
                    )
                ],
                bulk=Bulk(
                    src_id="bulk_id",
                    formula="XYZ",
                    elements=["X", "Y", "Z"],
                    other_fields={"extra_bulk_field": "extra_bulk_value"},
                ),
                slab=Slab(
                    atoms=Atoms(
                        cell=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)),
                        pbc=(True, True, True),
                        numbers=[1],
                        positions=[(1.1, 1.2, 1.3)],
                        tags=[0],
                        other_fields={"extra_slab_field": "extra_slab_value"},
                    ),
                    metadata=SlabMetadata(
                        bulk_src_id="bulk_id",
                        millers=(1, 1, 1),
                        shift=0.25,
                        top=False,
                        other_fields={"extra_meta_field": "extra_meta_value"},
                    ),
                ),
                model="test_model",
                ephemeral=True,
                adsorbate_reaction="A + B -> C",
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "adsorbate": "ABC",
    "adsorbate_configs": [
        {
            "cell": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            "pbc": [true, false, true],
            "numbers": [1, 2],
            "positions": [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
            "tags": [2, 2],
            "extra_ad_field": "extra_ad_value"
        }
    ],
    "bulk": {
        "src_id": "bulk_id",
        "formula": "XYZ",
        "els": ["X", "Y", "Z"],
        "extra_bulk_field": "extra_bulk_value"
    },
    "slab": {
        "slab_atomsobject": {
            "cell": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            "pbc": [true, true, true],
            "numbers": [1],
            "positions": [[1.1, 1.2, 1.3]],
            "tags": [0],
            "extra_slab_field": "extra_slab_value"
        },
        "slab_metadata": {
            "bulk_id": "bulk_id",
            "millers": [1, 1, 1],
            "shift": 0.25,
            "top": false,
            "extra_meta_field": "extra_meta_value"
        }
    },
    "model": "test_model",
    "ephemeral": true,
    "adsorbate_reaction": "A + B -> C",
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )


class TestAdsorbateSlabRelaxationsRequest_req_fields_only(
    ModelTestWrapper.ModelTest[AdsorbateSlabRelaxationsRequest]
):
    """
    Serde tests for the AdsorbateSlabRelaxationsRequest data model in which
    optional fields are omitted.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=AdsorbateSlabRelaxationsRequest(
                adsorbate="ABC",
                adsorbate_configs=[
                    Atoms(
                        cell=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)),
                        pbc=(True, False, True),
                        numbers=[1, 2],
                        positions=[(1.1, 1.2, 1.3), (1.4, 1.5, 1.6)],
                        tags=[2, 2],
                    )
                ],
                bulk=Bulk(
                    src_id="bulk_id",
                    formula="XYZ",
                    elements=["X", "Y", "Z"],
                ),
                slab=Slab(
                    atoms=Atoms(
                        cell=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)),
                        pbc=(True, True, True),
                        numbers=[1],
                        positions=[(1.1, 1.2, 1.3)],
                        tags=[0],
                    ),
                    metadata=SlabMetadata(
                        bulk_src_id="bulk_id",
                        millers=(1, 1, 1),
                        shift=0.25,
                        top=False,
                    ),
                ),
                model="test_model",
            ),
            obj_json="""
{
    "adsorbate": "ABC",
    "adsorbate_configs": [
        {
            "cell": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            "pbc": [true, false, true],
            "numbers": [1, 2],
            "positions": [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
            "tags": [2, 2]
        }
    ],
    "bulk": {
        "src_id": "bulk_id",
        "formula": "XYZ",
        "els": ["X", "Y", "Z"]
    },
    "slab": {
        "slab_atomsobject": {
            "cell": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            "pbc": [true, true, true],
            "numbers": [1],
            "positions": [[1.1, 1.2, 1.3]],
            "tags": [0]
        },
        "slab_metadata": {
            "bulk_id": "bulk_id",
            "millers": [1, 1, 1],
            "shift": 0.25,
            "top": false
        }
    },
    "model": "test_model"
}
""",
            *args,
            **kwargs,
        )


class TestAdsorbateSlabRelaxationResult(
    ModelTestWrapper.ModelTest[AdsorbateSlabRelaxationResult]
):
    """
    Serde tests for the AdsorbateSlabRelaxationResult data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=AdsorbateSlabRelaxationResult(
                config_id=1,
                status=Status.SUCCESS,
                system_id="sys_id",
                cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                pbc=(True, False, True),
                numbers=[1, 2],
                positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                tags=[0, 1],
                energy=100.1,
                energy_trajectory=[99.9, 100.1],
                forces=[(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "config_id": 1,
    "status": "success",
    "system_id": "sys_id",
    "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
    "pbc": [true, false, true],
    "numbers": [1, 2],
    "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
    "tags": [0, 1],
    "energy": 100.1,
    "energy_trajectory": [99.9, 100.1],
    "forces": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )

    def test_to_ase_atoms(self) -> None:
        @dataclass
        class TestCase:
            message: str
            result: AdsorbateSlabRelaxationResult
            expected_atoms: ASEAtoms
            expected_energy: Union[float, Type[Exception]]
            expected_forces: Union[np.ndarray, Type[Exception]]
            expected_unconstrained_forces: Union[np.ndarray, Type[Exception]]

        # Helper function to construct an ase.Atoms object with the
        # input attributes.
        def get_ase_atoms(
            energy: Optional[float],
            forces: Optional[List[Tuple[float, float, float]]],
            **kwargs: Any,
        ) -> ASEAtoms:
            atoms = ASEAtoms(**kwargs)
            atoms.calc = SinglePointCalculator(
                atoms=atoms,
                energy=energy,
                forces=forces,
            )
            return atoms

        test_cases: List[TestCase] = [
            # If all optional fields are omitted, the generated ase.Atoms
            # object should also have no values set
            TestCase(
                message="optional fields omitted",
                result=AdsorbateSlabRelaxationResult(
                    config_id=1,
                    status=Status.SUCCESS,
                ),
                expected_atoms=ASEAtoms(),
                expected_energy=Exception,
                expected_forces=Exception,
                expected_unconstrained_forces=Exception,
            ),
            # If all fields are included, the generated ase.Atoms object
            # should have positions, forces, etc. configured
            TestCase(
                message="all fields set",
                result=AdsorbateSlabRelaxationResult(
                    config_id=1,
                    status=Status.SUCCESS,
                    system_id="sys_id",
                    cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                    pbc=(True, False, True),
                    numbers=[1, 2],
                    positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                    tags=[0, 1],
                    energy=100.1,
                    energy_trajectory=[99.9, 100.1],
                    forces=[(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
                ),
                expected_atoms=get_ase_atoms(
                    cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                    pbc=(True, False, True),
                    numbers=[1, 2],
                    positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                    tags=[0, 1],
                    energy=100.1,
                    forces=[(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
                ),
                expected_energy=100.1,
                # The constraint on the first atom causes its force to be
                # zeroed out
                expected_forces=np.array([(0, 0, 0), (0.4, 0.5, 0.6)], float),
                expected_unconstrained_forces=np.array(
                    [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)], float
                ),
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                # Check that the atoms object is constructed correctly
                ase_atoms = case.result.to_ase_atoms()
                self.assertEqual(ase_atoms, case.expected_atoms)

                # Check the energy (or that an exception is raised if expected)
                if isinstance(case.expected_energy, float):
                    self.assertEqual(
                        case.expected_energy,
                        ase_atoms.get_potential_energy(),
                    )
                else:
                    with self.assertRaises(case.expected_energy):
                        ase_atoms.get_potential_energy()

                # Check the forces (or that an exception is raised if expected)
                if isinstance(case.expected_forces, np.ndarray):
                    self.assertEqual(
                        case.expected_forces.tolist(),
                        ase_atoms.get_forces().tolist(),
                    )
                else:
                    with self.assertRaises(case.expected_forces):
                        ase_atoms.get_forces()

                # Check the unconstrained forces (or that an exception is
                # raised if expected)
                if isinstance(case.expected_unconstrained_forces, np.ndarray):
                    self.assertEqual(
                        case.expected_unconstrained_forces.tolist(),
                        ase_atoms.get_forces(apply_constraint=False).tolist(),
                    )
                else:
                    with self.assertRaises(case.expected_unconstrained_forces):
                        ase_atoms.get_forces(apply_constraint=False)


class TestAdsorbateSlabRelaxationResult_req_fields_only(
    ModelTestWrapper.ModelTest[AdsorbateSlabRelaxationResult]
):
    """
    Serde tests for the AdsorbateSlabRelaxationResult data model in which
    optional fields are omitted.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=AdsorbateSlabRelaxationResult(
                config_id=1,
                status=Status.SUCCESS,
            ),
            obj_json="""
{
    "config_id": 1,
    "status": "success"
}
""",
            *args,
            **kwargs,
        )


class TestAdsorbateSlabRelaxationsResults(
    ModelTestWrapper.ModelTest[AdsorbateSlabRelaxationsResults]
):
    """
    Serde tests for the AdsorbateSlabRelaxationsResults data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=AdsorbateSlabRelaxationsResults(
                configs=[
                    AdsorbateSlabRelaxationResult(
                        config_id=1,
                        status=Status.SUCCESS,
                        system_id="sys_id",
                        cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                        pbc=(True, False, True),
                        numbers=[1, 2],
                        positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                        tags=[0, 1],
                        energy=100.1,
                        energy_trajectory=[99.9, 100.1],
                        forces=[(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
                        other_fields={"extra_adslab_field": "extra_adslab_value"},
                    )
                ],
                omitted_config_ids=[1, 2, 3],
                other_fields={"extra_field": "extra_value"},
            ),
            obj_json="""
{
    "configs": [{
        "config_id": 1,
        "status": "success",
        "system_id": "sys_id",
        "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
        "pbc": [true, false, true],
        "numbers": [1, 2],
        "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
        "tags": [0, 1],
        "energy": 100.1,
        "energy_trajectory": [99.9, 100.1],
        "forces": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "extra_adslab_field": "extra_adslab_value"
    }],
    "omitted_config_ids": [1, 2, 3],
    "extra_field": "extra_value"
}
""",
            *args,
            **kwargs,
        )
