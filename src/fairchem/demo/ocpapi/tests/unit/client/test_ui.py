from dataclasses import dataclass
from typing import List, Optional
from unittest import TestCase as UnitTestCase

from fairchem.demo.ocpapi.client import get_results_ui_url


class TestUI(UnitTestCase):
    def test_get_results_ui_url(self) -> None:
        @dataclass
        class TestCase:
            message: str
            api_host: str
            system_id: str
            expected: Optional[str]

        test_cases: List[TestCase] = [
            # If the prod host is used, then a URL to the prod UI
            # should be returned
            TestCase(
                message="prod host",
                api_host="open-catalyst-api.metademolab.com",
                system_id="abc",
                expected="https://open-catalyst.metademolab.com/results/abc",
            ),
            # If an unknown host name is used, then no URL should be returned
            TestCase(
                message="unknown host",
                api_host="unknown.host",
                system_id="abc",
                expected=None,
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                actual = get_results_ui_url(case.api_host, case.system_id)
                self.assertEqual(case.expected, actual)
