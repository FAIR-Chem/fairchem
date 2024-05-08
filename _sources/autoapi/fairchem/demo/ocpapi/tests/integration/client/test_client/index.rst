:py:mod:`fairchem.demo.ocpapi.tests.integration.client.test_client`
===================================================================

.. py:module:: fairchem.demo.ocpapi.tests.integration.client.test_client


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.demo.ocpapi.tests.integration.client.test_client.TestClient



Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.demo.ocpapi.tests.integration.client.test_client._ensure_system_deleted



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.demo.ocpapi.tests.integration.client.test_client.log


.. py:data:: log

   

.. py:function:: _ensure_system_deleted(client: fairchem.demo.ocpapi.client.Client, system_id: str) -> AsyncGenerator[None, None]
   :async:

   Immediately yields control to the caller. When control returns to this
   function, try to delete the system with the input id.


.. py:class:: TestClient(methodName='runTest')


   Bases: :py:obj:`unittest.IsolatedAsyncioTestCase`

   Tests that calls to a real server are handled correctly.

   .. py:attribute:: CLIENT
      :type: fairchem.demo.ocpapi.client.Client

      

   .. py:attribute:: KNOWN_SYSTEM_ID
      :type: str
      :value: 'f9eacd8f-748c-41dd-ae43-f263dd36d735'

      

   .. py:method:: test_get_models() -> None
      :async:


   .. py:method:: test_get_bulks() -> None
      :async:


   .. py:method:: test_get_adsorbates() -> None
      :async:


   .. py:method:: test_get_slabs() -> None
      :async:


   .. py:method:: test_get_adsorbate_slab_configs() -> None
      :async:


   .. py:method:: test_submit_adsorbate_slab_relaxations__gemnet_oc() -> None
      :async:


   .. py:method:: test_submit_adsorbate_slab_relaxations__equiformer_v2() -> None
      :async:


   .. py:method:: test_get_adsorbate_slab_relaxations_request() -> None
      :async:


   .. py:method:: test_get_adsorbate_slab_relaxations_results__all_fields_and_configs() -> None
      :async:


   .. py:method:: test_get_adsorbate_slab_relaxations_results__limited_fields_and_configs() -> None
      :async:



