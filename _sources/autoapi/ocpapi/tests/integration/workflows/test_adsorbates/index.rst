:py:mod:`ocpapi.tests.integration.workflows.test_adsorbates`
============================================================

.. py:module:: ocpapi.tests.integration.workflows.test_adsorbates


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpapi.tests.integration.workflows.test_adsorbates.TestAdsorbates




.. py:class:: TestAdsorbates(methodName='runTest')


   Bases: :py:obj:`unittest.IsolatedAsyncioTestCase`

   Tests that workflow methods run against a real server execute correctly.

   .. py:attribute:: CLIENT
      :type: fairchem.demo.ocpapi.client.Client

      

   .. py:attribute:: KNOWN_SYSTEM_ID
      :type: str
      :value: 'f9eacd8f-748c-41dd-ae43-f263dd36d735'

      

   .. py:method:: test_get_adsorbate_slab_relaxation_results() -> None
      :async:


   .. py:method:: test_wait_for_adsorbate_slab_relaxations() -> None
      :async:


   .. py:method:: test_find_adsorbate_binding_sites() -> None
      :async:



