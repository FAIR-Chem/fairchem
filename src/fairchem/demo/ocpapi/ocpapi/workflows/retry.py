import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from tenacity import RetryCallState
from tenacity import retry as tenacity_retry
from tenacity import (
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    stop_never,
    wait_fixed,
    wait_random,
)
from tenacity.wait import wait_base

from ocpapi.client import (
    NonRetryableRequestException,
    RateLimitExceededException,
    RequestException,
)


@dataclass
class RateLimitLogging:
    """
    Controls logging when rate limits are hit.
    """

    logger: logging.Logger
    """
    The logger to use.
    """

    action: str
    """
    A short description of the action being attempted.
    """


class _wait_check_retry_after(wait_base):
    """
    Tenacity wait strategy that first checks whether RateLimitExceededException
    was raised and that it includes a retry-after value; if so wait, for that
    amount of time. Otherwise, fall back to the provided default strategy.
    """

    def __init__(
        self,
        default_wait: wait_base,
        rate_limit_logging: Optional[RateLimitLogging] = None,
    ) -> None:
        """
        Args:
            default_wait: If a retry-after value was not provided in an API
                response, use this wait method.
            rate_limit_logging: If not None, log statements will be generated
                using this configuration when a rate limit is hit.
        """
        self._default_wait = default_wait
        self._rate_limit_logging = rate_limit_logging

    def __call__(self, retry_state: RetryCallState) -> float:
        """
        If a RateLimitExceededException was raised and has a retry_after value,
        return it. Otherwise use the default waiter method.
        """
        exception = retry_state.outcome.exception()
        if isinstance(exception, RateLimitExceededException):
            if exception.retry_after is not None:
                # Log information about the rate limit if needed
                wait_for: float = exception.retry_after.total_seconds()
                if (l := self._rate_limit_logging) is not None:
                    l.logger.info(
                        f"Request to {l.action} was rate limited with "
                        f"retry-after = {wait_for} seconds"
                    )
                return wait_for
        return self._default_wait(retry_state)


NoLimitType = Literal[0]
NO_LIMIT: NoLimitType = 0


def retry_api_calls(
    max_attempts: Union[int, NoLimitType] = 3,
    rate_limit_logging: Optional[RateLimitLogging] = None,
    fixed_wait_sec: float = 2,
    max_jitter_sec: float = 1,
) -> Any:
    """
    Decorator with sensible defaults for retrying calls to the OCP API.

    Args:
        max_attempts: The maximum number of calls to make. If NO_LIMIT,
            retries will be made forever.
        rate_limit_logging: If not None, log statements will be generated
            using this configuration when a rate limit is hit.
        fixed_wait_sec: The fixed number of seconds to wait when retrying an
            exception that does *not* include a retry-after value. The default
            value is sensible; this is exposed mostly for testing.
        max_jitter_sec: The maximum number of seconds that will be randomly
            added to wait times. The default value is sensible; this is exposed
            mostly for testing.
    """
    return tenacity_retry(
        # Retry forever if no limit was applied. Otherwise stop after the
        # max number of attempts has been made.
        stop=stop_never
        if max_attempts == NO_LIMIT
        else stop_after_attempt(max_attempts),
        # If the API returns that a rate limit was breached and gives a
        # retry-after value, use that. Otherwise wait a fixed number of
        # seconds. In all cases, add a random jitter.
        wait=_wait_check_retry_after(
            wait_fixed(fixed_wait_sec),
            rate_limit_logging,
        )
        + wait_random(0, max_jitter_sec),
        # Retry any API exceptions unless they are explicitly marked as
        # not retryable.
        retry=retry_if_exception_type(RequestException)
        & retry_if_not_exception_type(NonRetryableRequestException),
        # Raise the original exception instead of wrapping it in a
        # tenacity exception
        reraise=True,
    )
