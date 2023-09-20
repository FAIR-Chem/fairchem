from typing import Any, Literal, Union

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


class _wait_check_retry_after(wait_base):
    """
    Tenacity wait strategy that first checks whether RateLimitExceededException
    was raised and that it includes a retry-after value; if so wait, for that
    amount of time. Otherwise, fall back to the provided default strategy.
    """

    def __init__(self, default_wait: wait_base) -> None:
        """
        Args:
            default_wait: If a retry-after value was not provided in an API
                response, use this wait method.
        """
        self._default_wait = default_wait

    def __call__(self, retry_state: RetryCallState) -> float:
        """
        If a RateLimitExceededException was raised and has a retry_after value,
        return it. Otherwise use the default waiter method.
        """
        exception = retry_state.outcome.exception()
        if isinstance(exception, RateLimitExceededException):
            if exception.retry_after is not None:
                return exception.retry_after.total_seconds()
        return self._default_wait(retry_state)


NoLimitType = Literal[0]
NO_LIMIT: NoLimitType = 0


def retry_api_calls(max_attempts: Union[int, NoLimitType] = 3) -> Any:
    """
    Decorator with sensible defaults for retrying calls to the OCP API.

    Args:
        max_attempts: The maximum number of calls to make. If NO_LIMIT,
            retries will be made forever.
    """
    return tenacity_retry(
        # Retry forever if no limit was applied. Otherwise stop after the
        # max number of attempts has been made.
        stop=stop_never
        if max_attempts == NO_LIMIT
        else stop_after_attempt(max_attempts),
        # If the API returns that a rate limit was breached and gives a
        # retry-after value, use that. Otherwise wait 2 seconds. In all
        # cases, add a random jitter.
        wait=_wait_check_retry_after(wait_fixed(2)) + wait_random(0, 1),
        # Retry any API exceptions unless they are explicitly marked as
        # not retryable.
        retry=retry_if_exception_type(RequestException)
        & retry_if_not_exception_type(NonRetryableRequestException),
    )
