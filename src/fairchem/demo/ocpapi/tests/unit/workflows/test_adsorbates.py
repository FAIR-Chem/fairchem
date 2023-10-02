from dataclasses import dataclass
from typing import Any, Final, Iterable, List, Optional, Tuple, Type, Union
from unittest import mock, IsolatedAsyncioTestCase, TestCase as UnitTestCase
from ocpapi.client import (
    AdsorbateSlabRelaxationResult,
    AdsorbateSlabRelaxationsResults,
    Atoms,
    Client,
    RequestException,
    Slab,
    SlabMetadata,
    Status,
)
from ocpapi.workflows import (
    get_adsorbate_slab_relaxation_results,
    keep_slabs_with_miller_indices,
)


class MockGetRelaxationResults:
    """
    Helper that can be used to mock calls to
    Client.get_adsorbate_slab_relaxations_results(). This allows for
    some configs to be returned with "success" status and others with
    "omitted", similar to the behavior in the API.
    """

    def __init__(
        self,
        num_configs: int,
        max_configs_to_return: int,
        raise_on_first_call: Optional[Exception] = None,
    ) -> None:
        self._num_configs = num_configs
        self._max_configs_to_return = max_configs_to_return
        self._to_raise = raise_on_first_call

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

        # Return success for all config ids up to the max number allowed
        # to be returned. Return omitted for the rest.
        num_success = self._max_configs_to_return
        return AdsorbateSlabRelaxationsResults(
            configs=[
                AdsorbateSlabRelaxationResult(
                    config_id=i,
                    status=Status.SUCCESS,
                )
                for i in config_ids[:num_success]
            ]
            + [
                AdsorbateSlabRelaxationResult(
                    config_id=i,
                    status=Status.OMITTED,
                )
                for i in config_ids[num_success:]
            ]
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
                        ]
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
                            AdsorbateSlabRelaxationResult(
                                config_id=2,
                                status=Status.OMITTED,
                            ),
                            AdsorbateSlabRelaxationResult(
                                config_id=3,
                                status=Status.OMITTED,
                            ),
                            AdsorbateSlabRelaxationResult(
                                config_id=4,
                                status=Status.OMITTED,
                            ),
                        ]
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
                            AdsorbateSlabRelaxationResult(
                                config_id=4,
                                status=Status.OMITTED,
                            ),
                        ]
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
                    raise_on_first_call=Exception(),
                ),
                config_ids=None,
                expected_exception=Exception,
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
        pass

    async def test_find_adsorbate_binding_sites(self) -> None:
        pass

    def test_keep_slabs_with_miller_indices(self) -> None:
        @dataclass
        class TestCase:
            message: str
            allowed_miller_indices: Iterable[Tuple[int, int, int]]
            input_slabs: Iterable[Slab]
            expected: List[bool]

        # Helper function that generates a slab with the input
        # miller indices
        def new_slab(miller_indices: Tuple[int, int, int]) -> Slab:
            return Slab(
                atoms=Atoms(
                    cell=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                    pbc=(True, True, True),
                    numbers=[],
                    positions=[],
                    tags=[],
                ),
                metadata=SlabMetadata(
                    bulk_src_id="",
                    millers=miller_indices,
                    shift=0,
                    top=True,
                ),
            )

        test_cases: List[TestCase] = [
            # If no miller indices are defined, then no slabs should be kept
            TestCase(
                message="no miller indices allowed",
                allowed_miller_indices=[],
                input_slabs=[
                    new_slab((1, 0, 0)),
                    new_slab((1, 1, 0)),
                    new_slab((1, 1, 1)),
                ],
                expected=[False, False, False],
            ),
            # If no slabs are defined then nothing should be returned
            TestCase(
                message="no slabs provided",
                allowed_miller_indices=[(1, 1, 1)],
                input_slabs=[],
                expected=[],
            ),
            # Any miller indices that do match should be kept
            TestCase(
                message="some miller indices matched",
                allowed_miller_indices=[
                    (1, 0, 1),  # Won't match anything
                    (1, 0, 0),  # Will match
                    (1, 1, 1),  # Will match
                ],
                input_slabs=[
                    new_slab((1, 0, 0)),
                    new_slab((1, 1, 0)),
                    new_slab((1, 1, 1)),
                ],
                expected=[True, False, True],
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                slab_filter = keep_slabs_with_miller_indices(
                    case.allowed_miller_indices
                )
                results = [slab_filter(slab) for slab in case.input_slabs]
                self.assertEqual(results, case.expected)
