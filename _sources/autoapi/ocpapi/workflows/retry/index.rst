ocpapi.workflows.retry
======================

.. py:module:: ocpapi.workflows.retry


Attributes
----------

.. autoapisummary::

   ocpapi.workflows.retry.NoLimitType
   ocpapi.workflows.retry.NO_LIMIT


Classes
-------

.. autoapisummary::

   ocpapi.workflows.retry.RateLimitLogging
   ocpapi.workflows.retry._wait_check_retry_after


Functions
---------

.. autoapisummary::

   ocpapi.workflows.retry.retry_api_calls


Module Contents
---------------

.. py:class:: RateLimitLogging

   Controls logging when rate limits are hit.


   .. py:attribute:: logger
      :type:  logging.Logger

      The logger to use.


   .. py:attribute:: action
      :type:  str

      A short description of the action being attempted.


.. py:class:: _wait_check_retry_after(default_wait: tenacity.wait.wait_base, rate_limit_logging: RateLimitLogging | None = None)

   Bases: :py:obj:`tenacity.wait.wait_base`


   Tenacity wait strategy that first checks whether RateLimitExceededException
   was raised and that it includes a retry-after value; if so wait, for that
   amount of time. Otherwise, fall back to the provided default strategy.


   .. py:method:: __call__(retry_state: tenacity.RetryCallState) -> float

      If a RateLimitExceededException was raised and has a retry_after value,
      return it. Otherwise use the default waiter method.



.. py:data:: NoLimitType

.. py:data:: NO_LIMIT
   :type:  NoLimitType
   :value: 0


.. py:function:: retry_api_calls(max_attempts: int | NoLimitType = 3, rate_limit_logging: RateLimitLogging | None = None, fixed_wait_sec: float = 2, max_jitter_sec: float = 1) -> Any

   Decorator with sensible defaults for retrying calls to the OCP API.

   :param max_attempts: The maximum number of calls to make. If NO_LIMIT,
                        retries will be made forever.
   :param rate_limit_logging: If not None, log statements will be generated
                              using this configuration when a rate limit is hit.
   :param fixed_wait_sec: The fixed number of seconds to wait when retrying an
                          exception that does *not* include a retry-after value. The default
                          value is sensible; this is exposed mostly for testing.
   :param max_jitter_sec: The maximum number of seconds that will be randomly
                          added to wait times. The default value is sensible; this is exposed
                          mostly for testing.


