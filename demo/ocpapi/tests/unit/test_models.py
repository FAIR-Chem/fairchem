import json
from dataclasses import dataclass
from typing import Any, Final, Generic, List, Optional, Type, TypeVar
from unittest import TestCase as UnitTestCase

from ocpapi.models import AdsorbatesResponse, Bulk, BulksResponse, _Model

T = TypeVar("T", bound=_Model)


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


class TestBulksResponse(ModelTestWrapper.ModelTest[BulksResponse]):
    """
    Serde tests for the BulksResponse data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=BulksResponse(
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


class TestAdsorbatesResponse(ModelTestWrapper.ModelTest[AdsorbatesResponse]):
    """
    Serde tests for the AdsorbatesResponse data model.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            obj=AdsorbatesResponse(
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
