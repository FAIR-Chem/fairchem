import itertools
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import partial
from typing import Any, Dict, Final, Iterable, List, Optional, Type, Union
from unittest import IsolatedAsyncioTestCase
from unittest import TestCase as UnitTestCase
from unittest import mock

from fairchem.demo.ocpapi.client import (
    Adsorbates,
    AdsorbateSlabConfigs,
    AdsorbateSlabRelaxationResult,
    AdsorbateSlabRelaxationsResults,
    AdsorbateSlabRelaxationsSystem,
    Atoms,
    Bulk,
    Bulks,
    Client,
    Model,
    Models,
    RateLimitExceededException,
    RequestException,
    Slab,
    SlabMetadata,
    Slabs,
    Status,
)
from fairchem.demo.ocpapi.client.ui import _API_TO_UI_HOSTS
from fairchem.demo.ocpapi.workflows import (
    AdsorbateBindingSites,
    AdsorbateSlabRelaxations,
    UnsupportedAdsorbateException,
    UnsupportedBulkException,
    UnsupportedModelException,
    find_adsorbate_binding_sites,
    get_adsorbate_slab_relaxation_results,
    keep_all_slabs,
    keep_slabs_with_miller_indices,
    wait_for_adsorbate_slab_relaxations,
)


# Exception used in test cases below
class TestException(Exception):
    __test__ = False


class MockGetRelaxationResults:
    """
    Helper that can be used to mock calls to
    Client.get_adsorbate_slab_relaxations_results(). This allows for
    some configs to be returned with "success" status and others to be
    omitted, similar to the behavior in the API.
    """

    def __init__(
        self,
        num_configs: int,
        max_configs_to_return: int,
        status_to_return: Optional[Iterable[Status]] = None,
        raise_on_first_call: Optional[Exception] = None,
    ) -> None:
        self._num_configs = num_configs
        self._max_configs_to_return = max_configs_to_return
        self._to_raise = raise_on_first_call
        if status_to_return:
            self._status_to_return = iter(status_to_return)
        else:
            self._status_to_return = itertools.repeat(Status.SUCCESS)

    def __call__(
        self,
        *args: Any,
        config_ids: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> AdsorbateSlabRelaxationsResults:
        # If an exception is expected, then raise it
        if self._to_raise is not None:
            to_raise = self._to_raise
            # So the exception is not raised again on the next call
            # to this method
            self._to_raise = None
            raise to_raise

        # If no config IDs were requested, prepare to return results
        # for all configurations
        if config_ids is None:
            config_ids = list(range(self._num_configs))

        # Return status for all config ids up to the max number allowed
        # to be returned. Return omitted for the rest.
        status: Status = next(self._status_to_return)
        num_to_return: int = self._max_configs_to_return
        return AdsorbateSlabRelaxationsResults(
            configs=[
                AdsorbateSlabRelaxationResult(
                    config_id=i,
                    status=status,
                )
                for i in config_ids[:num_to_return]
            ],
            omitted_config_ids=config_ids[num_to_return:],
        )


class TestMockGetRelaxationResults(UnitTestCase):
    # Tests the MockGetRelaxationResults to make sure it works as expected
    # before using it below
    def test___call__(self) -> None:
        @dataclass
        class TestCase:
            message: str
            mock_results: MockGetRelaxationResults
            config_ids: Optional[List[int]]
            expected: List[
                Union[
                    AdsorbateSlabRelaxationsResults,
                    Type[Exception],
                ]
            ]

        test_cases: List[TestCase] = [
            # If an exception is passed to the constructor, then it should
            # be raised only on the first call
            TestCase(
                message="exception on first call",
                mock_results=MockGetRelaxationResults(
                    num_configs=1,
                    max_configs_to_return=1,
                    raise_on_first_call=Exception(),
                ),
                config_ids=[0],
                expected=[
                    Exception,
                    AdsorbateSlabRelaxationsResults(
                        configs=[
                            AdsorbateSlabRelaxationResult(
                                config_id=0,
                                status=Status.SUCCESS,
                            )
                        ],
                        omitted_config_ids=[],
                    ),
                ],
            ),
            # If no config ids are passed to the call, then the number of
            # max_configs_to_return should have success and the result should
            # be omitted.
            TestCase(
                message="all config ids",
                mock_results=MockGetRelaxationResults(
                    num_configs=5,
                    max_configs_to_return=2,
                ),
                config_ids=None,
                expected=[
                    AdsorbateSlabRelaxationsResults(
                        configs=[
                            AdsorbateSlabRelaxationResult(
                                config_id=0,
                                status=Status.SUCCESS,
                            ),
                            AdsorbateSlabRelaxationResult(
                                config_id=1,
                                status=Status.SUCCESS,
                            ),
                        ],
                        omitted_config_ids=[2, 3, 4],
                    ),
                ],
            ),
            # If config ids are passed to the call, only those that are
            # requested should be returned, and any number above
            # max_configs_to_return should be omitted.
            TestCase(
                message="subset of config ids",
                mock_results=MockGetRelaxationResults(
                    num_configs=5,
                    max_configs_to_return=2,
                ),
                config_ids=[2, 3, 4],
                expected=[
                    AdsorbateSlabRelaxationsResults(
                        configs=[
                            AdsorbateSlabRelaxationResult(
                                config_id=2,
                                status=Status.SUCCESS,
                            ),
                            AdsorbateSlabRelaxationResult(
                                config_id=3,
                                status=Status.SUCCESS,
                            ),
                        ],
                        omitted_config_ids=[4],
                    ),
                ],
            ),
            # If statuses are passed to the call, they should be returned
            # in the defined order.
            TestCase(
                message="custom statuses",
                mock_results=MockGetRelaxationResults(
                    num_configs=2,
                    max_configs_to_return=2,
                    status_to_return=[
                        Status.SUCCESS,
                        Status.DOES_NOT_EXIST,
                        Status.FAILED_RELAXATION,
                        Status.NOT_AVAILABLE,
                    ],
                ),
                config_ids=None,
                expected=[
                    AdsorbateSlabRelaxationsResults(
                        configs=[
                            AdsorbateSlabRelaxationResult(
                                config_id=0,
                                status=Status.SUCCESS,
                            ),
                            AdsorbateSlabRelaxationResult(
                                config_id=1,
                                status=Status.SUCCESS,
                            ),
                        ],
                    ),
                    AdsorbateSlabRelaxationsResults(
                        configs=[
                            AdsorbateSlabRelaxationResult(
                                config_id=0,
                                status=Status.DOES_NOT_EXIST,
                            ),
                            AdsorbateSlabRelaxationResult(
                                config_id=1,
                                status=Status.DOES_NOT_EXIST,
                            ),
                        ],
                    ),
                    AdsorbateSlabRelaxationsResults(
                        configs=[
                            AdsorbateSlabRelaxationResult(
                                config_id=0,
                                status=Status.FAILED_RELAXATION,
                            ),
                            AdsorbateSlabRelaxationResult(
                                config_id=1,
                                status=Status.FAILED_RELAXATION,
                            ),
                        ],
                    ),
                    AdsorbateSlabRelaxationsResults(
                        configs=[
                            AdsorbateSlabRelaxationResult(
                                config_id=0,
                                status=Status.NOT_AVAILABLE,
                            ),
                            AdsorbateSlabRelaxationResult(
                                config_id=1,
                                status=Status.NOT_AVAILABLE,
                            ),
                        ],
                    ),
                ],
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                # Call the obj as many times as there are expected results so
                # we can check each one
                for expected in case.expected:
                    # If a response is expected, make sure it is returned
                    if isinstance(expected, AdsorbateSlabRelaxationsResults):
                        result = case.mock_results(config_ids=case.config_ids)
                        self.assertEqual(result, expected)

                    # Otherwise make sure the expected exception is raised
                    else:
                        with self.assertRaises(expected):
                            case.mock_results(config_ids=case.config_ids)


class TestAdsorbates(IsolatedAsyncioTestCase):
    async def test_get_adsorbate_slab_relaxation_results(self) -> None:
        # This test uses a mocked API client. Mapping API responses to client
        # results is tested elsewhere. Here we only want to ensure that the
        # get_adsorbate_slab_relaxation_results() method correctly retries
        # client requests.
        @dataclass
        class TestCase:
            message: str
            mock_results: MockGetRelaxationResults
            config_ids: Optional[List[int]]
            expected: Final[Optional[List[AdsorbateSlabRelaxationResult]]] = None
            expected_exception: Final[Optional[Type[Exception]]] = None

        test_cases: List[TestCase] = [
            # If a non-retryable exception is raised from the client, it should
            # be re-raised
            TestCase(
                message="no retryable exception",
                mock_results=MockGetRelaxationResults(
                    num_configs=1,
                    max_configs_to_return=1,
                    raise_on_first_call=TestException(),
                ),
                config_ids=None,
                expected_exception=TestException,
            ),
            # If a retryable exception is raised from the client, it should be
            # retried and results eventually returned
            TestCase(
                message="retry exception when possible",
                mock_results=MockGetRelaxationResults(
                    num_configs=1,
                    max_configs_to_return=1,
                    raise_on_first_call=RequestException("", "", ""),
                ),
                config_ids=None,
                expected=[
                    AdsorbateSlabRelaxationResult(
                        config_id=0,
                        status=Status.SUCCESS,
                    )
                ],
            ),
            # If all configs are requested, they should all be returned, even
            # some are initially omitted by the API
            TestCase(
                message="retry fetching on all configs",
                mock_results=MockGetRelaxationResults(
                    num_configs=2,
                    # Return "omitted" for all but 1 config on each attempt
                    max_configs_to_return=1,
                ),
                config_ids=None,
                expected=[
                    AdsorbateSlabRelaxationResult(
                        config_id=0,
                        status=Status.SUCCESS,
                    ),
                    AdsorbateSlabRelaxationResult(
                        config_id=1,
                        status=Status.SUCCESS,
                    ),
                ],
            ),
            # If a subset of configs are requested, they should all be
            # returned, even some are initially omitted by the API
            TestCase(
                message="retry fetching on subset of configs",
                mock_results=MockGetRelaxationResults(
                    num_configs=3,
                    # Return "omitted" for all but 1 config on each attempt
                    max_configs_to_return=1,
                ),
                config_ids=[1, 2],
                expected=[
                    AdsorbateSlabRelaxationResult(
                        config_id=1,
                        status=Status.SUCCESS,
                    ),
                    AdsorbateSlabRelaxationResult(
                        config_id=2,
                        status=Status.SUCCESS,
                    ),
                ],
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                # Mock the client
                client = mock.create_autospec(Client)
                client.get_adsorbate_slab_relaxations_results.side_effect = (
                    case.mock_results
                )

                # Coroutine that will fetch results
                coro = get_adsorbate_slab_relaxation_results(
                    system_id="not used",
                    config_ids=case.config_ids,
                    client=client,
                )

                # Make sure an exception is raised if expected
                if case.expected_exception is not None:
                    with self.assertRaises(case.expected_exception):
                        await coro

                # Otherwise make sure the expected result is returned
                else:
                    result = await coro
                    self.assertEqual(case.expected, result)

    async def test_wait_for_adsorbate_slab_relaxations(self) -> None:
        @dataclass
        class TestCase:
            message: str
            mock_results: MockGetRelaxationResults
            expected: Final[Optional[Dict[int, Status]]] = None
            expected_exception: Final[Optional[Type[Exception]]] = None

        test_cases: List[TestCase] = [
            # If a non-retryable exception is raised from the client, it should
            # be re-raised
            TestCase(
                message="no retryable exception",
                mock_results=MockGetRelaxationResults(
                    num_configs=1,
                    max_configs_to_return=1,
                    raise_on_first_call=TestException(),
                ),
                expected_exception=TestException,
            ),
            # If a retryable exception is raised from the client, it should be
            # retried and results eventually returned
            TestCase(
                message="retry exception when possible",
                mock_results=MockGetRelaxationResults(
                    num_configs=1,
                    max_configs_to_return=1,
                    raise_on_first_call=RequestException("", "", ""),
                ),
                expected={0: Status.SUCCESS},
            ),
            # Calls should be retried until all statuses are collected and in
            # a terminal state
            TestCase(
                message="retry until all available",
                mock_results=MockGetRelaxationResults(
                    # Each time get_adsorbate_slab_relaxation_results is called,
                    # 3 calls to Client.get_adsorbate_slab_relaxations_results()
                    # should be made since each will only return 1 config
                    num_configs=3,
                    max_configs_to_return=1,
                    status_to_return=[
                        # The first calls should be retried since not available
                        Status.NOT_AVAILABLE,
                        Status.NOT_AVAILABLE,
                        Status.NOT_AVAILABLE,
                        # The rest of the calls should return a terminal status
                        Status.SUCCESS,
                        Status.FAILED_RELAXATION,
                        Status.DOES_NOT_EXIST,
                    ],
                ),
                expected={
                    0: Status.SUCCESS,
                    1: Status.FAILED_RELAXATION,
                    2: Status.DOES_NOT_EXIST,
                },
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                # Mock the client
                client = mock.create_autospec(Client)
                client.get_adsorbate_slab_relaxations_results.side_effect = (
                    case.mock_results
                )

                # Coroutine that will fetch results
                coro = wait_for_adsorbate_slab_relaxations(
                    system_id="not used",
                    check_immediately=True,
                    slow_interval_sec=0.01,
                    fast_interval_sec=0.01,
                    client=client,
                )

                # Make sure an exception is raised if expected
                if case.expected_exception is not None:
                    with self.assertRaises(case.expected_exception):
                        await coro

                # Otherwise make sure the expected result is returned
                else:
                    result = await coro
                    self.assertEqual(case.expected, result)

    async def test_find_adsorbate_binding_sites(self) -> None:
        @dataclass
        class TestCase:
            message: str
            adsorbate: str
            bulk: str
            client_get_models: List[Union[Models, Exception]]
            client_get_adsorbates: List[Union[Adsorbates, Exception]]
            client_get_bulks: List[Union[Bulks, Exception]]
            client_get_slabs: List[Union[Slabs, Exception]]
            client_get_adsorbate_slab_configs: List[
                Union[AdsorbateSlabConfigs, Exception]
            ]
            client_submit_adsorbate_slab_relaxations: List[
                Union[AdsorbateSlabRelaxationsSystem, Exception],
            ]
            client_get_adsorbate_slab_relaxations_results: List[
                Union[AdsorbateSlabRelaxationsResults, Exception]
            ]
            non_default_args: Optional[Dict[str, Any]] = None
            expected: Final[Optional[AdsorbateBindingSites]] = None
            expected_exception: Final[Optional[Type[Exception]]] = None

        # List of models to return from the API
        models: Models = Models(
            models=[
                Model(id="equiformer_v2_31M_s2ef_all_md"),
                Model(id="model_2"),
            ]
        )

        # List of adsorbates to return from the API
        adsorbates: Adsorbates = Adsorbates(
            adsorbates_supported=["*A", "*B"],
        )

        # List of bulks to return from the API
        bulk_1: Bulk = Bulk(
            src_id="id-1",
            formula="AB",
            elements=["A", "B"],
        )
        bulk_2: Bulk = Bulk(
            src_id="id_2",
            formula="CD",
            elements=["C", "D"],
        )
        bulks: Bulks = Bulks(bulks_supported=[bulk_1, bulk_2])

        # List of slabs to return from the API
        slab_1: Slab = Slab(
            atoms=Atoms(
                cell=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                pbc=[True, True, False],
                numbers=[100, 101],
                positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
                tags=[0, 0],
            ),
            metadata=SlabMetadata(
                bulk_src_id="id-1",
                millers=(1, 0, 0),
                shift=0.1,
                top=True,
            ),
        )
        slab_2: Slab = Slab(
            atoms=Atoms(
                cell=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                pbc=[True, True, False],
                numbers=[100, 101],
                positions=[(0.1, 0.1, 0.1), (0.6, 0.6, 0.6)],
                tags=[0, 0],
            ),
            metadata=SlabMetadata(
                bulk_src_id="id-1",
                millers=(1, 1, 1),
                shift=0.5,
                top=False,
            ),
        )
        slabs: Slabs = Slabs(slabs=[slab_1, slab_2])

        # Adslab configurations to return from the API
        adsorbate_config_1: Atoms = Atoms(
            cell=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            pbc=[True, True, False],
            numbers=[102],
            positions=[(0.1, 0.1, 0.1)],
            tags=[2],
        )
        adsorbate_config_2: Atoms = Atoms(
            cell=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            pbc=[True, True, False],
            numbers=[102],
            positions=[(0.2, 0.2, 0.2)],
            tags=[2],
        )
        adsorbate_slab_configs_1: AdsorbateSlabConfigs = AdsorbateSlabConfigs(
            adsorbate_configs=[adsorbate_config_1, adsorbate_config_2],
            slab=Slab(  # Slab 1
                atoms=replace(slab_1.atoms, tags=[0, 1]),
                metadata=slab_1.metadata,
            ),
        )
        adsorbate_slab_configs_2: AdsorbateSlabConfigs = AdsorbateSlabConfigs(
            adsorbate_configs=[adsorbate_config_1, adsorbate_config_2],
            slab=Slab(  # Slab 2
                atoms=replace(slab_2.atoms, tags=[0, 1]),
                metadata=slab_2.metadata,
            ),
        )

        # Responses from the API when submitting adslab relaxations
        system_1: AdsorbateSlabRelaxationsSystem = AdsorbateSlabRelaxationsSystem(
            system_id="ABC",
            config_ids=[0, 1],
        )
        system_2: AdsorbateSlabRelaxationsSystem = AdsorbateSlabRelaxationsSystem(
            system_id="XYZ",
            config_ids=[0, 1],
        )

        # Responses from the API when fetching relaxation results
        results: AdsorbateSlabRelaxationsResults = AdsorbateSlabRelaxationsResults(
            configs=[
                AdsorbateSlabRelaxationResult(
                    config_id=0,
                    status=Status.SUCCESS,
                ),
                AdsorbateSlabRelaxationResult(
                    config_id=1,
                    status=Status.SUCCESS,
                ),
            ],
        )

        # Use a real host name as the mocked host. This won't be called, but
        # it ensures that the UI URL will be generated correctly.
        api_host, ui_host = next(iter(_API_TO_UI_HOSTS.items()))

        test_cases: List[TestCase] = [
            # An exception raised when fetching models should be re-raised
            TestCase(
                message="exception while getting models",
                adsorbate="*B",
                bulk="id-1",
                client_get_models=[TestException()],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[bulks],
                client_get_slabs=[slabs],
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                expected_exception=TestException,
            ),
            # If the requested model is not supported in the API then an
            # exception should be raised
            TestCase(
                message="model not supported",
                adsorbate="*B",
                bulk="id-1",
                non_default_args={
                    "model": "model_3",  # Not is set returned by API
                },
                client_get_models=[models],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[bulks],
                client_get_slabs=[slabs],
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                expected_exception=UnsupportedModelException,
            ),
            # An exception raised when fetching adsorbates should be re-raised
            TestCase(
                message="exception while getting adsorbates",
                adsorbate="*B",
                bulk="id-1",
                client_get_models=[models],
                client_get_adsorbates=[TestException()],
                client_get_bulks=[bulks],
                client_get_slabs=[slabs],
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                expected_exception=TestException,
            ),
            # If the adsorbate is not supported then an exception should be
            # raised
            TestCase(
                message="adsorbate not supported",
                adsorbate="*C",  # Not is set returned by API
                bulk="id-1",
                client_get_models=[models],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[bulks],
                client_get_slabs=[slabs],
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                expected_exception=UnsupportedAdsorbateException,
            ),
            # An exception raised when fetching bulks should be re-raised
            TestCase(
                message="exception while getting bulks",
                adsorbate="*B",
                bulk="id-1",
                client_get_models=[models],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[TestException()],
                client_get_slabs=[slabs],
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                expected_exception=TestException,
            ),
            # If the bulk is not supported then an exception should be raised
            TestCase(
                message="bulk not supported",
                adsorbate="*B",
                bulk="id-3",  # Not in set returned by API
                client_get_models=[models],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[bulks],
                client_get_slabs=[slabs],
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                expected_exception=UnsupportedBulkException,
            ),
            # An exception raised when fetching slabs should be re-raised
            TestCase(
                message="exception while getting slabs",
                adsorbate="*B",
                bulk="id-1",
                client_get_models=[models],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[bulks],
                client_get_slabs=[TestException()],
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                expected_exception=TestException,
            ),
            # An exception raised when fetching adsorbate slab configs
            # should be re-raised
            TestCase(
                message="exception while getting adslab configs",
                adsorbate="*B",
                bulk="id-1",
                client_get_models=[models],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[bulks],
                client_get_slabs=[slabs],
                client_get_adsorbate_slab_configs=[
                    TestException(),
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                expected_exception=TestException,
            ),
            # An exception raised when submitting relaxations should be
            # re-raised
            TestCase(
                message="exception while submitting relaxations",
                adsorbate="*B",
                bulk="id-1",
                client_get_models=[models],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[bulks],
                client_get_slabs=[slabs],
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    TestException(),
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                expected_exception=TestException,
            ),
            # An exception raised while fetching relaxation results should
            # be re-raised
            TestCase(
                message="exception while getting relaxation results",
                adsorbate="*B",
                bulk="id-1",
                client_get_models=[models],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[bulks],
                client_get_slabs=[slabs],
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    TestException(),
                    results,
                    results,
                    results,
                ],
                expected_exception=TestException,
            ),
            # If no slabs are generated, the results that are returned
            # should have an empty slabs list
            TestCase(
                message="no slabs",
                adsorbate="*B",
                bulk="id-1",
                client_get_models=[models],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[bulks],
                client_get_slabs=[Slabs(slabs=[])],  # No slabs generated
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[system_1, system_2],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                expected=AdsorbateBindingSites(
                    adsorbate="*B",
                    bulk=bulk_1,
                    model="equiformer_v2_31M_s2ef_all_md",
                    slabs=[],
                ),
            ),
            # Retryable exceptions on all routes should be retried and all
            # results should be collected eventually
            TestCase(
                message="retryable exceptions",
                adsorbate="*B",
                bulk="id-1",
                client_get_models=[RequestException("", "", ""), models],
                client_get_adsorbates=[RequestException("", "", ""), adsorbates],
                client_get_bulks=[RequestException("", "", ""), bulks],
                client_get_slabs=[RequestException("", "", ""), slabs],
                client_get_adsorbate_slab_configs=[
                    RequestException("", "", ""),
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    RequestException("", "", ""),
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    RequestException("", "", ""),
                    RateLimitExceededException("", "", timedelta(seconds=0.01)),
                    results,
                    results,
                    results,
                    results,
                ],
                expected=AdsorbateBindingSites(
                    adsorbate="*B",
                    bulk=bulk_1,
                    model="equiformer_v2_31M_s2ef_all_md",
                    slabs=[
                        AdsorbateSlabRelaxations(
                            slab=adsorbate_slab_configs_1.slab,
                            system_id=system_1.system_id,
                            api_host=api_host,
                            ui_url=f"https://{ui_host}/results/{system_1.system_id}",
                            configs=results.configs,
                        ),
                        AdsorbateSlabRelaxations(
                            slab=adsorbate_slab_configs_2.slab,
                            system_id=system_2.system_id,
                            api_host=api_host,
                            ui_url=f"https://{ui_host}/results/{system_2.system_id}",
                            configs=results.configs,
                        ),
                    ],
                ),
            ),
            # Non-default values should be used where configured
            TestCase(
                message="non-default values",
                adsorbate="*B",
                bulk="id-1",
                client_get_models=[models],
                client_get_adsorbates=[adsorbates],
                client_get_bulks=[bulks],
                client_get_slabs=[slabs],
                client_get_adsorbate_slab_configs=[
                    adsorbate_slab_configs_1,
                    adsorbate_slab_configs_2,
                ],
                client_submit_adsorbate_slab_relaxations=[
                    system_1,
                    system_2,
                ],
                client_get_adsorbate_slab_relaxations_results=[
                    results,
                    results,
                    results,
                    results,
                ],
                non_default_args={
                    "model": "model_2",
                    "adslab_filter": keep_slabs_with_miller_indices([(1, 0, 0)]),
                },
                expected=AdsorbateBindingSites(
                    adsorbate="*B",
                    bulk=bulk_1,
                    model="model_2",
                    slabs=[
                        AdsorbateSlabRelaxations(
                            slab=adsorbate_slab_configs_1.slab,
                            system_id=system_1.system_id,
                            api_host=api_host,
                            ui_url=f"https://{ui_host}/results/{system_1.system_id}",
                            configs=results.configs,
                        ),
                    ],
                ),
            ),
        ]

        for case in test_cases:
            with ExitStack() as es:
                es.enter_context(self.subTest(msg=case.message))

                # Mock the client
                client = mock.create_autospec(Client)
                client.get_models.side_effect = case.client_get_models
                client.get_adsorbates.side_effect = case.client_get_adsorbates
                client.get_bulks.side_effect = case.client_get_bulks
                client.get_slabs.side_effect = case.client_get_slabs
                client.get_adsorbate_slab_configs.side_effect = (
                    case.client_get_adsorbate_slab_configs
                )
                client.submit_adsorbate_slab_relaxations.side_effect = (
                    case.client_submit_adsorbate_slab_relaxations
                )
                client.get_adsorbate_slab_relaxations_results.side_effect = (
                    case.client_get_adsorbate_slab_relaxations_results
                )
                type(client).host = mock.PropertyMock(return_value=api_host)

                # Change the timeouts between calls to get relaxation results.
                # We don't need to test here that we wait tens of seconds when
                # results aren't ready - we just want to ensure that we do
                # retry until they are fetched successfully.
                es.enter_context(
                    mock.patch(
                        "ocpapi.workflows.adsorbates.wait_for_adsorbate_slab_relaxations",
                        partial(
                            wait_for_adsorbate_slab_relaxations,
                            slow_interval_sec=0.01,
                            fast_interval_sec=0.01,
                            pbar=None,
                        ),
                    )
                )

                # Coroutine that will fetch results
                other = case.non_default_args if case.non_default_args else {}
                if "adslab_filter" not in other:
                    # Override default that will prompt for input
                    other["adslab_filter"] = keep_all_slabs()
                coro = find_adsorbate_binding_sites(
                    adsorbate=case.adsorbate,
                    bulk=case.bulk,
                    client=client,
                    **other,
                )

                # Make sure an exception is raised if expected
                if case.expected_exception is not None:
                    with self.assertRaises(case.expected_exception):
                        await coro

                # Otherwise make sure the expected result is returned
                else:
                    result = await coro
                    # Results are fetched in coroutines and the order is not
                    # guaranteed. Sort both the expected and actual results so
                    # this next assertion doesn't fail randomly
                    result.slabs = sorted(
                        result.slabs,
                        key=lambda x: x.system_id,
                    )
                    case.expected.slabs = sorted(
                        case.expected.slabs,
                        key=lambda x: x.system_id,
                    )
                    self.assertEqual(case.expected, result)
