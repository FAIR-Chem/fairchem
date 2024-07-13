ocpapi.workflows.filter
=======================

.. py:module:: ocpapi.workflows.filter


Classes
-------

.. autoapisummary::

   ocpapi.workflows.filter.keep_all_slabs
   ocpapi.workflows.filter.keep_slabs_with_miller_indices
   ocpapi.workflows.filter.prompt_for_slabs_to_keep


Module Contents
---------------

.. py:class:: keep_all_slabs

   Adslab filter than returns all slabs.


   .. py:method:: __call__(adslabs: list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]) -> list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]
      :async:



.. py:class:: keep_slabs_with_miller_indices(miller_indices: Iterable[tuple[int, int, int]])

   Adslab filter that keeps any slabs with the configured miller indices.
   Slabs with other miller indices will be ignored.


   .. py:method:: __call__(adslabs: list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]) -> list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]
      :async:



.. py:class:: prompt_for_slabs_to_keep

   Adslab filter than presents the user with an interactive prompt to choose
   which of the input slabs to keep.


   .. py:method:: _sort_key(adslab: fairchem.demo.ocpapi.client.AdsorbateSlabConfigs) -> tuple[tuple[int, int, int], float, str]
      :staticmethod:


      Generates a sort key from the input adslab. Returns the miller indices,
      shift, and top/bottom label so that they will be sorted by those values
      in that order.



   .. py:method:: __call__(adslabs: list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]) -> list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]
      :async:



