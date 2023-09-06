import json
from dataclasses import dataclass
from typing import Any, Generic, List, Type, TypeVar
from unittest import TestCase as UnitTestCase

from ocpapi.models import Bulk, BulksResponse, _Model

T = TypeVar("T", bound=_Model)


class ModelTestWrapper:
    class ModelTest(UnitTestCase, Generic[T]):
        """
        Base class for all tests below that assert behavior of data models.
        """

        def __init__(
            self,
            *args: Any,
            model_type: Type[T],
            default_obj: T,
            default_json: str,
            complete_obj: T,
            complete_json: str,
            **kwargs: Any,
        ) -> None:
            """
            Args:
                model_type: Subclass of _Model that is being tested.
                default_obj: A model instance that has been created with
                    default values.
                default_json: JSON-serialized version of default_obj.
                complete_obj: A model instance in which all fields, even
                    unknown ones, are included.
                complete_json: JSON-serialized version of complete_obj.
            """
            super().__init__(*args, **kwargs)
            self._model_type = model_type
            self._default_obj = default_obj
            self._default_json = default_json
            self._complete_obj = complete_obj
            self._complete_json = complete_json

        def test_from_json(self) -> None:
            @dataclass
            class TestCase:
                message: str
                json_repr: str
                expected: T

            test_cases: List[TestCase] = [
                # If the json object is empty then default values should
                # be used for all fields
                TestCase(
                    message="empty object",
                    json_repr="{}",
                    expected=self._default_obj,
                ),
                # If all fields are set then they should be included in the
                # resulting object
                TestCase(
                    message="all fields set",
                    json_repr=self._complete_json,
                    expected=self._complete_obj,
                ),
            ]

            for case in test_cases:
                with self.subTest(msg=case.message):
                    actual = self._model_type.from_json(case.json_repr)
                    self.assertEqual(actual, case.expected)

        def test_to_json(self) -> None:
            @dataclass
            class TestCase:
                message: str
                obj: T
                expected: str

            test_cases: List[TestCase] = [
                # An empty model instance should serialize default values
                TestCase(
                    message="empty object",
                    obj=self._model_type(),
                    expected=self._default_json,
                ),
                # All explicitly-defined fields should serialize
                TestCase(
                    message="all fields set",
                    obj=self._complete_obj,
                    expected=self._complete_json,
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
            model_type=Bulk,
            default_obj=Bulk(
                src_id="",
                formula="",
                elements=[],
                other_fields={},
            ),
            default_json="""
{
    "src_id": "",
    "els": [],
    "formula": ""
}
""",
            complete_obj=Bulk(
                src_id="test_id",
                elements=["A", "B"],
                formula="AB2",
                other_fields={"extra_field": "extra_value"},
            ),
            complete_json="""
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
            model_type=BulksResponse,
            default_obj=BulksResponse(
                bulks_supported=[],
                other_fields={},
            ),
            default_json="""
{
    "bulks_supported": []
}
""",
            complete_obj=BulksResponse(
                bulks_supported=[
                    Bulk(
                        src_id="test_id",
                        elements=["A", "B"],
                        formula="AB2",
                    )
                ],
                other_fields={"extra_field": "extra_value"},
            ),
            complete_json="""
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
