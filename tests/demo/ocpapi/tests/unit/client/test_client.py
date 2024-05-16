from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union
from unittest import IsolatedAsyncioTestCase

import responses

from fairchem.demo.ocpapi.client import (
    Adsorbates,
    AdsorbateSlabConfigs,
    AdsorbateSlabRelaxationResult,
    AdsorbateSlabRelaxationsRequest,
    AdsorbateSlabRelaxationsResults,
    AdsorbateSlabRelaxationsSystem,
    Atoms,
    Bulk,
    Bulks,
    Client,
    Model,
    Models,
    NonRetryableRequestException,
    RateLimitExceededException,
    RequestException,
    Slab,
    SlabMetadata,
    Slabs,
    Status,
)
from fairchem.demo.ocpapi.client.models import _DataModel


class TestClient(IsolatedAsyncioTestCase):
    """
    Tests with mocked responses to ensure that they are handled correctly.
    """

    async def _run_common_tests_against_route(
        self,
        method: str,
        route: str,
        client_method_name: str,
        successful_response_code: int,
        successful_response_body: str,
        successful_response_object: Optional[_DataModel],
        client_method_args: Optional[Dict[str, Any]] = None,
        expected_request_params: Optional[Dict[str, Any]] = None,
        expected_request_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        @dataclass
        class TestCase:
            message: str
            scheme: str
            host: str
            response_body: Union[str, Exception]
            response_code: int
            response_headers: Optional[Dict[str, str]] = None
            expected: Optional[_DataModel] = None
            expected_request_params: Optional[Dict[str, Any]] = None
            expected_request_body: Optional[Dict[str, Any]] = None
            expected_exception: Optional[Exception] = None

        test_cases: List[TestCase] = [
            # If a 429 response code is returned, then a
            # RateLimitExceededException should be raised
            TestCase(
                message="rate limit exceeded",
                scheme="https",
                host="test_host",
                response_body='{"message": "failed"}',
                response_code=429,
                response_headers={"Retry-After": "100"},
                expected_request_params=expected_request_params,
                expected_request_body=expected_request_body,
                expected_exception=RateLimitExceededException(
                    method=method,
                    url=f"https://test_host/{route}",
                    retry_after=timedelta(seconds=100),
                ),
            ),
            # If a 429 response code is returned, then a
            # RateLimitExceededException should be raised - ensure correct
            # handling when retry-after header is not present
            TestCase(
                message="rate limit exceeded, no retry-after",
                scheme="https",
                host="test_host",
                response_body='{"message": "failed"}',
                response_code=429,
                response_headers={},
                expected_request_params=expected_request_params,
                expected_request_body=expected_request_body,
                expected_exception=RateLimitExceededException(
                    method=method,
                    url=f"https://test_host/{route}",
                    retry_after=None,
                ),
            ),
            # If a 400-level response code is returned then a
            # NonRetryableRequestException should be raised
            TestCase(
                message="non-retryable error",
                scheme="https",
                host="test_host",
                response_body='{"message": "failed"}',
                response_code=404,
                response_headers={},
                expected_request_params=expected_request_params,
                expected_request_body=expected_request_body,
                expected_exception=NonRetryableRequestException(
                    method=method,
                    url=f"https://test_host/{route}",
                    cause=(
                        "Unexpected response code: 404. "
                        'Response body: {"message": "failed"}'
                    ),
                ),
            ),
            # If another unexpected response code is returned then an exception
            # should be raised
            TestCase(
                message="non-200 response code",
                scheme="https",
                host="test_host",
                response_body='{"message": "failed"}',
                response_code=500,
                expected_request_params=expected_request_params,
                expected_request_body=expected_request_body,
                expected_exception=RequestException(
                    method=method,
                    url=f"https://test_host/{route}",
                    cause=(
                        "Unexpected response code: 500. "
                        'Response body: {"message": "failed"}'
                    ),
                ),
            ),
            # If an exception is raised from within requests, it should be
            # re-raised in the client
            TestCase(
                message="exception in request handling",
                scheme="https",
                host="test_host",
                # This tells the responses library to raise an exception
                response_body=Exception("exception message"),
                response_code=successful_response_code,
                expected_request_params=expected_request_params,
                expected_request_body=expected_request_body,
                expected_exception=RequestException(
                    method=method,
                    url=f"https://test_host/{route}",
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
                scheme="https",
                host="test_host",
                response_body=successful_response_body,
                response_code=successful_response_code,
                expected=successful_response_object,
                expected_request_params=expected_request_params,
                expected_request_body=expected_request_body,
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                # Match the request body if one is expected
                match = []
                if case.expected_request_body is not None:
                    match.append(
                        responses.matchers.json_params_matcher(
                            case.expected_request_body
                        )
                    )
                if case.expected_request_params is not None:
                    match.append(
                        responses.matchers.query_param_matcher(
                            case.expected_request_params
                        )
                    )

                # Mock the response to the request in the current test case
                with responses.RequestsMock() as mock_responses:
                    mock_responses.add(
                        method,
                        f"{case.scheme}://{case.host}/{route}",
                        body=case.response_body,
                        headers=case.response_headers,
                        status=case.response_code,
                        match=match,
                    )

                    # Create the coroutine that will run the request
                    client = Client(scheme=case.scheme, host=case.host)
                    request_method = getattr(client, client_method_name)
                    args = client_method_args if client_method_args else {}
                    request_coro = request_method(**args)

                    # Ensure that an exception is raised if one is expected
                    if case.expected_exception is not None:
                        with self.assertRaises(type(case.expected_exception)) as ex:
                            await request_coro
                        self.assertEqual(
                            vars(case.expected_exception),
                            vars(ex.exception),
                        )
                        self.assertEqual(
                            str(case.expected_exception),
                            str(ex.exception),
                        )

                    # If an exception is not expected, then make sure the
                    # response is correct
                    else:
                        response = await request_coro
                        self.assertEqual(response, case.expected)

    def test_host(self) -> None:
        client = Client(host="test-host")
        self.assertEqual("test-host", client.host)

    async def test_get_models(self) -> None:
        await self._run_common_tests_against_route(
            method="GET",
            route="ocp/models",
            client_method_name="get_models",
            successful_response_code=200,
            successful_response_body="""
{
    "models": [
        {
            "id": "model_1"
        },
        {
            "id": "model_2"
        }
    ]
}
""",
            successful_response_object=Models(
                models=[
                    Model(id="model_1"),
                    Model(id="model_2"),
                ]
            ),
        )

    async def test_get_bulks(self) -> None:
        await self._run_common_tests_against_route(
            method="GET",
            route="ocp/bulks",
            client_method_name="get_bulks",
            successful_response_code=200,
            successful_response_body="""
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
            successful_response_object=Bulks(
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
        )

    async def test_get_adsorbates(self) -> None:
        await self._run_common_tests_against_route(
            method="GET",
            route="ocp/adsorbates",
            client_method_name="get_adsorbates",
            successful_response_code=200,
            successful_response_body="""
{
    "adsorbates_supported": ["A", "B"]
}
""",
            successful_response_object=Adsorbates(
                adsorbates_supported=["A", "B"],
            ),
        )

    async def test_get_slabs__bulk_by_id(self) -> None:
        await self._run_common_tests_against_route(
            method="POST",
            route="ocp/slabs",
            client_method_name="get_slabs",
            client_method_args={"bulk": "test_id"},
            expected_request_body={"bulk_src_id": "test_id"},
            successful_response_code=200,
            successful_response_body="""
{
    "slabs": [{
        "slab_atomsobject": {
            "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            "pbc": [true, false, true],
            "numbers": [1, 2],
            "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
            "tags": [0, 1]
        },
        "slab_metadata": {
            "bulk_id": "test_id",
            "millers": [-1, 0, 1],
            "shift": 0.25,
            "top": false
        }
    }]
}
""",
            successful_response_object=Slabs(
                slabs=[
                    Slab(
                        atoms=Atoms(
                            cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                            pbc=(True, False, True),
                            numbers=[1, 2],
                            positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                            tags=[0, 1],
                        ),
                        metadata=SlabMetadata(
                            bulk_src_id="test_id",
                            millers=(-1, 0, 1),
                            shift=0.25,
                            top=False,
                        ),
                    )
                ],
            ),
        )

    async def test_get_slabs__bulk_by_obj(self) -> None:
        await self._run_common_tests_against_route(
            method="POST",
            route="ocp/slabs",
            client_method_name="get_slabs",
            client_method_args={
                "bulk": Bulk(
                    src_id="test_id",
                    formula="AB",
                    elements=["A", "B"],
                )
            },
            expected_request_body={"bulk_src_id": "test_id"},
            successful_response_code=200,
            successful_response_body="""
{
    "slabs": [{
        "slab_atomsobject": {
            "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            "pbc": [true, false, true],
            "numbers": [1, 2],
            "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
            "tags": [0, 1]
        },
        "slab_metadata": {
            "bulk_id": "test_id",
            "millers": [-1, 0, 1],
            "shift": 0.25,
            "top": false
        }
    }]
}
""",
            successful_response_object=Slabs(
                slabs=[
                    Slab(
                        atoms=Atoms(
                            cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                            pbc=(True, False, True),
                            numbers=[1, 2],
                            positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                            tags=[0, 1],
                        ),
                        metadata=SlabMetadata(
                            bulk_src_id="test_id",
                            millers=(-1, 0, 1),
                            shift=0.25,
                            top=False,
                        ),
                    )
                ],
            ),
        )

    async def test_get_adsorbate_slab_configurations(self) -> None:
        await self._run_common_tests_against_route(
            method="POST",
            route="ocp/adsorbate-slab-configs",
            client_method_name="get_adsorbate_slab_configs",
            client_method_args={
                "adsorbate": "*A",
                "slab": Slab(
                    atoms=Atoms(
                        cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                        pbc=(True, False, True),
                        numbers=[1, 2],
                        positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                        tags=[0, 1],
                    ),
                    metadata=SlabMetadata(
                        bulk_src_id="test_id",
                        millers=(-1, 0, 1),
                        shift=0.25,
                        top=False,
                    ),
                ),
            },
            expected_request_body={
                "adsorbate": "*A",
                "slab": {
                    "slab_atomsobject": {
                        "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
                        "pbc": [True, False, True],
                        "numbers": [1, 2],
                        "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
                        "tags": [0, 1],
                    },
                    "slab_metadata": {
                        "bulk_id": "test_id",
                        "millers": [-1, 0, 1],
                        "shift": 0.25,
                        "top": False,
                    },
                },
            },
            successful_response_code=200,
            successful_response_body="""
{
    "adsorbate_configs": [
        {
            "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            "pbc": [true, false, true],
            "numbers": [1],
            "positions": [[1.1, 1.2, 1.3]],
            "tags": [2]
        }
    ],
    "slab": {
        "slab_atomsobject": {
            "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
            "pbc": [true, false, true],
            "numbers": [1, 2],
            "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
            "tags": [0, 1]
        },
        "slab_metadata": {
            "bulk_id": "test_id",
            "millers": [-1, 0, 1],
            "shift": 0.25,
            "top": false
        }
    }
}
""",
            successful_response_object=AdsorbateSlabConfigs(
                adsorbate_configs=[
                    Atoms(
                        cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                        pbc=(True, False, True),
                        numbers=[1],
                        positions=[(1.1, 1.2, 1.3)],
                        tags=[2],
                    )
                ],
                slab=Slab(
                    atoms=Atoms(
                        cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                        pbc=(True, False, True),
                        numbers=[1, 2],
                        positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                        tags=[0, 1],
                    ),
                    metadata=SlabMetadata(
                        bulk_src_id="test_id",
                        millers=(-1, 0, 1),
                        shift=0.25,
                        top=False,
                    ),
                ),
            ),
        )

    async def test_submit_adsorbate_slab_relaxations(self) -> None:
        await self._run_common_tests_against_route(
            method="POST",
            route="ocp/adsorbate-slab-relaxations",
            client_method_name="submit_adsorbate_slab_relaxations",
            client_method_args={
                "adsorbate": "*A",
                "adsorbate_configs": [
                    Atoms(
                        cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                        pbc=(True, False, True),
                        numbers=[1],
                        positions=[(1.1, 1.2, 1.3)],
                        tags=[2],
                    ),
                ],
                "bulk": Bulk(
                    src_id="test_id",
                    formula="AB",
                    elements=["A", "B"],
                ),
                "slab": Slab(
                    atoms=Atoms(
                        cell=((1.1, 2.1, 3.1), (4.1, 5.1, 6.1), (7.1, 8.1, 9.1)),
                        pbc=(True, False, True),
                        numbers=[1, 2],
                        positions=[(1.1, 1.2, 1.3), (2.1, 2.2, 2.3)],
                        tags=[0, 1],
                    ),
                    metadata=SlabMetadata(
                        bulk_src_id="test_id",
                        millers=(-1, 0, 1),
                        shift=0.25,
                        top=False,
                    ),
                ),
                "model": "test_model",
                "ephemeral": True,
            },
            expected_request_body={
                "adsorbate": "*A",
                "adsorbate_configs": [
                    {
                        "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
                        "pbc": [True, False, True],
                        "numbers": [1],
                        "positions": [[1.1, 1.2, 1.3]],
                        "tags": [2],
                    }
                ],
                "bulk": {
                    "src_id": "test_id",
                    "formula": "AB",
                    "els": ["A", "B"],
                },
                "slab": {
                    "slab_atomsobject": {
                        "cell": [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],
                        "pbc": [True, False, True],
                        "numbers": [1, 2],
                        "positions": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
                        "tags": [0, 1],
                    },
                    "slab_metadata": {
                        "bulk_id": "test_id",
                        "millers": [-1, 0, 1],
                        "shift": 0.25,
                        "top": False,
                    },
                },
                "model": "test_model",
                "ephemeral": True,
            },
            successful_response_code=200,
            successful_response_body="""
{
    "system_id": "sys_id",
    "config_ids": [1, 2, 3]
}
""",
            successful_response_object=AdsorbateSlabRelaxationsSystem(
                system_id="sys_id",
                config_ids=[1, 2, 3],
            ),
        )

    async def test_get_adsorbate_slab_relaxations_request(self) -> None:
        await self._run_common_tests_against_route(
            method="GET",
            route="ocp/adsorbate-slab-relaxations/test_system_id",
            client_method_name="get_adsorbate_slab_relaxations_request",
            client_method_args={"system_id": "test_system_id"},
            successful_response_code=200,
            successful_response_body="""
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
            successful_response_object=AdsorbateSlabRelaxationsRequest(
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
        )

    async def test_get_adsorbate_slab_relaxations_results__all_args(self) -> None:
        await self._run_common_tests_against_route(
            method="GET",
            route="ocp/adsorbate-slab-relaxations/test_sys_id/configs",
            client_method_name="get_adsorbate_slab_relaxations_results",
            client_method_args={
                "system_id": "test_sys_id",
                "config_ids": [1, 2],
                "fields": ["A", "B"],
            },
            expected_request_params={
                "config_id": ["1", "2"],
                "field": ["A", "B"],
            },
            successful_response_code=200,
            successful_response_body="""
{
    "configs": [
        {
            "config_id": 1,
            "status": "success"
        }
    ]
}
""",
            successful_response_object=AdsorbateSlabRelaxationsResults(
                configs=[
                    AdsorbateSlabRelaxationResult(
                        config_id=1,
                        status=Status.SUCCESS,
                    )
                ],
                omitted_config_ids=[],
            ),
        )

    async def test_get_adsorbate_slab_relaxations_results__req_args_only(self) -> None:
        await self._run_common_tests_against_route(
            method="GET",
            route="ocp/adsorbate-slab-relaxations/test_sys_id/configs",
            client_method_name="get_adsorbate_slab_relaxations_results",
            client_method_args={
                "system_id": "test_sys_id",
            },
            expected_request_params={},
            successful_response_code=200,
            successful_response_body="""
{
    "configs": [
        {
            "config_id": 1,
            "status": "success"
        }
    ]
}
""",
            successful_response_object=AdsorbateSlabRelaxationsResults(
                configs=[
                    AdsorbateSlabRelaxationResult(
                        config_id=1,
                        status=Status.SUCCESS,
                    )
                ],
                omitted_config_ids=[],
            ),
        )

    async def test_delete_adsorbate_slab_relaxations(self) -> None:
        await self._run_common_tests_against_route(
            method="DELETE",
            route="ocp/adsorbate-slab-relaxations/test_sys_id",
            client_method_name="delete_adsorbate_slab_relaxations",
            client_method_args={
                "system_id": "test_sys_id",
            },
            successful_response_code=200,
            successful_response_body="{}",
            successful_response_object=None,
        )
