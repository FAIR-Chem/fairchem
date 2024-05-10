:py:mod:`ocpapi.tests.integration.client.test_ui`
=================================================

.. py:module:: ocpapi.tests.integration.client.test_ui


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpapi.tests.integration.client.test_ui.TestUI




.. py:class:: TestUI(methodName='runTest')


   Bases: :py:obj:`unittest.TestCase`

   Tests that calls to a real server are handled correctly.

   .. py:attribute:: API_HOST
      :type: str
      :value: 'open-catalyst-api.metademolab.com'

      

   .. py:attribute:: KNOWN_SYSTEM_ID
      :type: str
      :value: 'f9eacd8f-748c-41dd-ae43-f263dd36d735'

      

   .. py:method:: test_get_results_ui_url() -> None



