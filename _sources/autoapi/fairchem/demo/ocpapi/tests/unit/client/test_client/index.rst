:py:mod:`fairchem.demo.ocpapi.tests.unit.client.test_client`
============================================================

.. py:module:: fairchem.demo.ocpapi.tests.unit.client.test_client


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.demo.ocpapi.tests.unit.client.test_client.TestClient




.. py:class:: TestClient(methodName='runTest')


   Bases: :py:obj:`unittest.IsolatedAsyncioTestCase`

   Tests with mocked responses to ensure that they are handled correctly.

   .. py:method:: _run_common_tests_against_route(method: str, route: str, client_method_name: str, successful_response_code: int, successful_response_body: str, successful_response_object: Optional[fairchem.demo.ocpapi.client.models._DataModel], client_method_args: Optional[Dict[str, Any]] = None, expected_request_params: Optional[Dict[str, Any]] = None, expected_request_body: Optional[Dict[str, Any]] = None) -> None
      :async:


   .. py:method:: test_host() -> None


   .. py:method:: test_get_models() -> None
      :async:


   .. py:method:: test_get_bulks() -> None
      :async:


   .. py:method:: test_get_adsorbates() -> None
      :async:


   .. py:method:: test_get_slabs__bulk_by_id() -> None
      :async:


   .. py:method:: test_get_slabs__bulk_by_obj() -> None
      :async:


   .. py:method:: test_get_adsorbate_slab_configurations() -> None
      :async:


   .. py:method:: test_submit_adsorbate_slab_relaxations() -> None
      :async:


   .. py:method:: test_get_adsorbate_slab_relaxations_request() -> None
      :async:


   .. py:method:: test_get_adsorbate_slab_relaxations_results__all_args() -> None
      :async:


   .. py:method:: test_get_adsorbate_slab_relaxations_results__req_args_only() -> None
      :async:


   .. py:method:: test_delete_adsorbate_slab_relaxations() -> None
      :async:



