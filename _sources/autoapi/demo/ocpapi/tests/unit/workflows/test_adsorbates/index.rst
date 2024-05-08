:py:mod:`demo.ocpapi.tests.unit.workflows.test_adsorbates`
==========================================================

.. py:module:: demo.ocpapi.tests.unit.workflows.test_adsorbates


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   demo.ocpapi.tests.unit.workflows.test_adsorbates.MockGetRelaxationResults
   demo.ocpapi.tests.unit.workflows.test_adsorbates.TestMockGetRelaxationResults
   demo.ocpapi.tests.unit.workflows.test_adsorbates.TestAdsorbates




.. py:exception:: TestException


   Bases: :py:obj:`Exception`

   Common base class for all non-exit exceptions.

   .. py:attribute:: __test__
      :value: False

      


.. py:class:: MockGetRelaxationResults(num_configs: int, max_configs_to_return: int, status_to_return: Optional[Iterable[fairchem.demo.ocpapi.client.Status]] = None, raise_on_first_call: Optional[Exception] = None)


   Helper that can be used to mock calls to
   Client.get_adsorbate_slab_relaxations_results(). This allows for
   some configs to be returned with "success" status and others to be
   omitted, similar to the behavior in the API.

   .. py:method:: __call__(*args: Any, config_ids: Optional[List[int]] = None, **kwargs: Any) -> fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationsResults



.. py:class:: TestMockGetRelaxationResults(methodName='runTest')


   Bases: :py:obj:`unittest.TestCase`

   A class whose instances are single test cases.

   By default, the test code itself should be placed in a method named
   'runTest'.

   If the fixture may be used for many test cases, create as
   many test methods as are needed. When instantiating such a TestCase
   subclass, specify in the constructor arguments the name of the test method
   that the instance is to execute.

   Test authors should subclass TestCase for their own tests. Construction
   and deconstruction of the test's environment ('fixture') can be
   implemented by overriding the 'setUp' and 'tearDown' methods respectively.

   If it is necessary to override the __init__ method, the base class
   __init__ method must always be called. It is important that subclasses
   should not change the signature of their __init__ method, since instances
   of the classes are instantiated automatically by parts of the framework
   in order to be run.

   When subclassing TestCase, you can set these attributes:
   * failureException: determines which exception will be raised when
       the instance's assertion methods fail; test methods raising this
       exception will be deemed to have 'failed' rather than 'errored'.
   * longMessage: determines whether long messages (including repr of
       objects used in assert methods) will be printed on failure in *addition*
       to any explicit message passed.
   * maxDiff: sets the maximum length of a diff in failure messages
       by assert methods using difflib. It is looked up as an instance
       attribute so can be configured by individual tests if required.

   .. py:method:: test___call__() -> None



.. py:class:: TestAdsorbates(methodName='runTest')


   Bases: :py:obj:`unittest.IsolatedAsyncioTestCase`

   A class whose instances are single test cases.

   By default, the test code itself should be placed in a method named
   'runTest'.

   If the fixture may be used for many test cases, create as
   many test methods as are needed. When instantiating such a TestCase
   subclass, specify in the constructor arguments the name of the test method
   that the instance is to execute.

   Test authors should subclass TestCase for their own tests. Construction
   and deconstruction of the test's environment ('fixture') can be
   implemented by overriding the 'setUp' and 'tearDown' methods respectively.

   If it is necessary to override the __init__ method, the base class
   __init__ method must always be called. It is important that subclasses
   should not change the signature of their __init__ method, since instances
   of the classes are instantiated automatically by parts of the framework
   in order to be run.

   When subclassing TestCase, you can set these attributes:
   * failureException: determines which exception will be raised when
       the instance's assertion methods fail; test methods raising this
       exception will be deemed to have 'failed' rather than 'errored'.
   * longMessage: determines whether long messages (including repr of
       objects used in assert methods) will be printed on failure in *addition*
       to any explicit message passed.
   * maxDiff: sets the maximum length of a diff in failure messages
       by assert methods using difflib. It is looked up as an instance
       attribute so can be configured by individual tests if required.

   .. py:method:: test_get_adsorbate_slab_relaxation_results() -> None
      :async:


   .. py:method:: test_wait_for_adsorbate_slab_relaxations() -> None
      :async:


   .. py:method:: test_find_adsorbate_binding_sites() -> None
      :async:



