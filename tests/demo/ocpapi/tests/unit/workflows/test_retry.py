from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import (
    Any,
    Callable,
    Final,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from unittest import TestCase as UnitTestCase
from unittest import mock

from fairchem.demo.ocpapi.client import (
    NonRetryableRequestException,
    RateLimitExceededException,
    RequestException,
)
from fairchem.demo.ocpapi.workflows import (
    NO_LIMIT,
    NoLimitType,
    RateLimitLogging,
    retry_api_calls,
)

T = TypeVar("T")


# Helper function that returns the input value immediately
def returns(val: T) -> Callable[[], T]:
    return lambda: val


# Helper function that raises the input exception
def raises(ex: Exception) -> Callable[[], None]:
    def func() -> None:
        raise ex

    return func


class TestRetry(UnitTestCase):
    def test_retry_api_calls__results(self) -> None:
        # Tests for retry behavior under various results (returning a
        # successful value, raising various exceptions, etc.)

        @dataclass
        class TestCase:
            message: str
            max_attempts: Union[int, NoLimitType]
            funcs: Iterable[Callable[[], Any]]
            expected_attempt_count: int
            expected_return_value: Final[Optional[Any]] = None
            expected_exception: Final[Optional[Type[Exception]]] = None

        test_cases: List[TestCase] = [
            # If a function runs successfully on the first call then exactly
            # one attempt should be made
            TestCase(
                message="success on first call",
                max_attempts=3,
                funcs=[returns(True)],
                expected_attempt_count=1,
                expected_return_value=True,
            ),
            # If a function raises a generic exception, it should never be
            # retried
            TestCase(
                message="non-api-type exception",
                max_attempts=3,
                funcs=[raises(Exception())],
                expected_attempt_count=1,
                expected_exception=Exception,
            ),
            # If a function raises an exception from the API that is not
            # retryable, then it should be re-raised
            TestCase(
                message="non-retryable api exception",
                max_attempts=3,
                funcs=[raises(NonRetryableRequestException("", "", ""))],
                expected_attempt_count=1,
                expected_exception=NonRetryableRequestException,
            ),
            # If a function raises an exception from the API that can be
            # retried, then another call should be made
            TestCase(
                message="retryable api exception, below max attempts",
                max_attempts=3,
                # Raise on the first attempt and return a value on the second
                funcs=[raises(RequestException("", "", "")), returns(True)],
                # Expect that two calls are made since the first should be
                # retried
                expected_attempt_count=2,
                expected_return_value=True,
            ),
            # If a function raises an exception from the API that can be
            # retried, but is raised more times than is allowed, then it
            # should be re-raised eventually
            TestCase(
                message="retryable api exception, exceeds max attempts",
                # Make at most two calls to the function
                max_attempts=2,
                # Raise on each attempt
                funcs=[
                    raises(RequestException("", "", "")),
                    raises(RequestException("", "", "")),
                ],
                # Expect that two calls are made since the first should be
                # retried
                expected_attempt_count=2,
                # Except that the exception is eventually raised
                expected_exception=RequestException,
            ),
            # If a function has no limit on the number of retries, then it
            # should retry the function until a non-retryable exception is
            # raised
            TestCase(
                message="no attempt limit",
                max_attempts=NO_LIMIT,
                # Raise several retryable exceptions before finally raising
                # one that cannot be retried
                funcs=[
                    raises(RequestException("", "", "")),
                    raises(RequestException("", "", "")),
                    raises(RequestException("", "", "")),
                    raises(RequestException("", "", "")),
                    raises(Exception()),
                ],
                expected_attempt_count=5,
                expected_exception=Exception,
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                func_iter = iter(case.funcs)

                # Retried function that runs all of the functions in the test
                # case
                @retry_api_calls(
                    max_attempts=case.max_attempts,
                    # Disable waits so that this tests run more quickly.
                    # Waits are tested elsewhere.
                    fixed_wait_sec=0,
                    max_jitter_sec=0,
                )
                def test_func() -> Any:
                    cur_func = next(func_iter)
                    return cur_func()

                # Make sure an exception is raised if one is expected
                if case.expected_exception is not None:
                    with self.assertRaises(case.expected_exception):
                        test_func()
                else:
                    return_value = test_func()
                    self.assertEqual(case.expected_return_value, return_value)

                # Make sure the number of function calls is expected
                self.assertEqual(
                    case.expected_attempt_count,
                    test_func.retry.statistics["attempt_number"],
                )

    def test_retry_api_calls__wait(self) -> None:
        # Tests for retry wait times

        @dataclass
        class TestCase:
            message: str
            funcs: Iterable[Callable[[], Any]]
            fixed_wait_sec: float
            max_jitter_sec: float
            expected_duration_range: Tuple[float, float]

        test_cases: List[TestCase] = [
            # A function that succeeds on the first attempt should return
            # without waiting
            TestCase(
                message="success on first call",
                funcs=[returns(None)],
                fixed_wait_sec=2,
                max_jitter_sec=1,
                # Function should return nearly immediately, but certainly
                # much faster than the fixed wait time
                expected_duration_range=(0, 0.1),
            ),
            # A function that raises a non-retryable exception should return
            # without waiting
            TestCase(
                message="non-retryable exception",
                funcs=[raises(Exception)],
                fixed_wait_sec=2,
                max_jitter_sec=1,
                # Function should return nearly immediately, but certainly
                # much faster than the fixed wait time
                expected_duration_range=(0, 0.1),
            ),
            # A function with a retryable API exception, but without a value
            # for retry-after, should wait based on the fixed wait time and
            # random jitter
            TestCase(
                message="retryable exception without retry-after",
                funcs=[raises(RequestException("", "", "")), returns(None)],
                fixed_wait_sec=0.2,
                max_jitter_sec=0.2,
                # Function should wait between 0.2 and 0.4 seconds - give a
                # small buffer on the upper bound
                expected_duration_range=(0.2, 0.5),
            ),
            # A function with a retryable API exception that includes a value
            # for retry-after should wait based on that returned time
            TestCase(
                message="retryable exception with retry-after",
                funcs=[
                    raises(RateLimitExceededException("", "", timedelta(seconds=0.2))),
                    returns(None),
                ],
                fixed_wait_sec=2,
                max_jitter_sec=0.2,
                # Function should wait between 0.2 and 0.4 seconds - give a
                # small buffer on the upper bound
                expected_duration_range=(0.2, 0.5),
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                func_iter = iter(case.funcs)

                # Retried function that runs all of the functions in the test
                # case
                @retry_api_calls(
                    fixed_wait_sec=case.fixed_wait_sec,
                    max_jitter_sec=case.max_jitter_sec,
                )
                def test_func() -> Any:
                    cur_func = next(func_iter)
                    return cur_func()

                # Time the function execution. Ignore any exceptions since
                # they may be intentionally raised in the test case.
                start = time.monotonic()
                with suppress(Exception):
                    test_func()
                took = time.monotonic() - start

                self.assertGreaterEqual(took, case.expected_duration_range[0])
                self.assertLessEqual(took, case.expected_duration_range[1])

    def test_retry_api_calls__logging(self) -> None:
        # Tests for logging during retries

        @dataclass
        class TestCase:
            message: str
            funcs: Iterable[Callable[[], Any]]
            log_action: str
            expected_log_statements: Iterable[str]

        test_cases: List[TestCase] = [
            # A function that succeeds immediately should not generate any
            # log statements
            TestCase(
                message="success on first call",
                funcs=[returns(None)],
                log_action="test_action",
                expected_log_statements=[],
            ),
            # A function that raises a non-retryable exception should not
            # generate any log statements
            TestCase(
                message="non-retryable exception",
                funcs=[raises(Exception)],
                log_action="test_action",
                expected_log_statements=[],
            ),
            # A function with a retryable API exception, but without a value
            # for retry-after, should not generate any log statements
            TestCase(
                message="base retryable exception",
                funcs=[raises(RequestException("", "", "")), returns(None)],
                log_action="test_action",
                expected_log_statements=[],
            ),
            # A function with a RateLimitExceededException exception, but
            # without a value for retry-after, should not generate any log
            # statements
            TestCase(
                message="rate limit exception without retry-after",
                funcs=[
                    raises(RateLimitExceededException("", "", None)),
                    returns(None),
                ],
                log_action="test_action",
                expected_log_statements=[],
            ),
            # A function with a RateLimitExceededException exception with a
            # value for retry-after should generate a log statement
            TestCase(
                message="rate limit exception with retry-after",
                funcs=[
                    raises(RateLimitExceededException("", "", timedelta(seconds=1))),
                    returns(None),
                ],
                log_action="test_action",
                expected_log_statements=[
                    (
                        "Request to test_action was rate limited with "
                        "retry-after = 1.0 seconds"
                    )
                ],
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                func_iter = iter(case.funcs)

                # Inject a mocked logger so we can intercept calls to log
                # messages
                import logging

                log = mock.create_autospec(logging.Logger)

                # Retried function that runs all of the functions in the test
                # case
                @retry_api_calls(
                    fixed_wait_sec=0,
                    max_jitter_sec=0,
                    rate_limit_logging=RateLimitLogging(
                        logger=log,
                        action=case.log_action,
                    ),
                )
                def test_func() -> Any:
                    cur_func = next(func_iter)
                    return cur_func()

                # Run the test function and make sure the excepted log
                # statements were generated. Ignore any exceptions since
                # they may be intentionally raised in the test case.
                with suppress(Exception):
                    test_func()
                log.info.assert_has_calls(
                    [mock.call(s) for s in case.expected_log_statements]
                )
