from dataclasses import dataclass
from typing import List, Optional, Union
from unittest import IsolatedAsyncioTestCase

import responses

from ocpapi.client import Client, RequestException
from ocpapi.models import Bulk, BulksResponse


class TestClient(IsolatedAsyncioTestCase):
    """
    Tests with mocked responses to ensure that they are handled correctly.
    """

    async def test_get_bulks(self) -> None:
        @dataclass
        class TestCase:
            message: str
            base_url: str
            response_body: Union[str, Exception]
            response_code: int
            expected: Optional[BulksResponse] = None
            expected_exception: Optional[Exception] = None

        test_cases: List[TestCase] = [
            # If a non-200 response code is returned then an exception should
            # be raised
            TestCase(
                message="non-200 response code",
                base_url="https://test_host/ocp",
                response_body='{"message": "failed"}',
                response_code=500,
                expected_exception=RequestException(
                    method="GET",
                    url="https://test_host/ocp/bulks",
                    cause=(
                        "Expected response code 200; got 500. "
                        'Body = {"message": "failed"}'
                    ),
                ),
            ),
            # If an exception is raised from within requests, it should be
            # re-raised in the client
            TestCase(
                message="exception in request handling",
                base_url="https://test_host/ocp",
                # This tells the responses library to raise an exception
                response_body=Exception("exception message"),
                response_code=200,
                expected_exception=RequestException(
                    method="GET",
                    url="https://test_host/ocp/bulks",
                    cause=(
                        "Exception while making request: "
                        "Exception: exception message"
                    ),
                ),
            ),
            # If the request is successful then data should be saved in
            # the response object
            TestCase(
                message="response with data",
                base_url="https://test_host/ocp",
                response_body="""
{
    "bulks_supported": [
        {
            "src_id": "1",
            "els": ["A", "B"],
            "formula": "AB2"
        },
        {
            "src_id": "2",
            "els": ["C"],
            "formula": "C60"
        }
    ]
}
""",
                response_code=200,
                expected=BulksResponse(
                    bulks_supported=[
                        Bulk(
                            src_id="1",
                            elements=["A", "B"],
                            formula="AB2",
                        ),
                        Bulk(
                            src_id="2",
                            elements=["C"],
                            formula="C60",
                        ),
                    ],
                ),
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                # Mock the response to the request in the current test case
                with responses.RequestsMock() as mock_responses:
                    mock_responses.add(
                        responses.GET,
                        f"{case.base_url}/bulks",
                        body=case.response_body,
                        status=case.response_code,
                    )

                    # Create the coroutine that will run the request
                    client = Client(case.base_url)
                    get_bulks_coro = client.get_bulks()

                    # Ensure that an exception is raised if one is expected
                    if case.expected_exception is not None:
                        with self.assertRaises(type(case.expected_exception)) as ex:
                            await get_bulks_coro
                        self.assertEqual(
                            str(case.expected_exception),
                            str(ex.exception),
                        )

                    # If an exception is not expected, then make sure the
                    # response is correct
                    else:
                        response = await get_bulks_coro
                        self.assertEqual(response, case.expected)
