ocpapi.client.ui
================

.. py:module:: ocpapi.client.ui


Attributes
----------

.. autoapisummary::

   ocpapi.client.ui._API_TO_UI_HOSTS


Functions
---------

.. autoapisummary::

   ocpapi.client.ui.get_results_ui_url


Module Contents
---------------

.. py:data:: _API_TO_UI_HOSTS
   :type:  Dict[str, str]

.. py:function:: get_results_ui_url(api_host: str, system_id: str) -> Optional[str]

   Generates the URL at which results for the input system can be
   visualized.

   :param api_host: The API host on which the system was run.
   :param system_id: ID of the system being visualized.

   :returns: The URL at which the input system can be visualized. None if the
             API host is not recognized.


