:py:mod:`fairchem.core.scripts.download_data`
=============================================

.. py:module:: fairchem.core.scripts.download_data


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.scripts.download_data.get_data
   fairchem.core.scripts.download_data.uncompress_data
   fairchem.core.scripts.download_data.preprocess_data
   fairchem.core.scripts.download_data.verify_count
   fairchem.core.scripts.download_data.cleanup



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.core.scripts.download_data.DOWNLOAD_LINKS_s2ef
   fairchem.core.scripts.download_data.DOWNLOAD_LINKS_is2re
   fairchem.core.scripts.download_data.S2EF_COUNTS
   fairchem.core.scripts.download_data.parser


.. py:data:: DOWNLOAD_LINKS_s2ef
   :type: dict[str, dict[str, str]]

   

.. py:data:: DOWNLOAD_LINKS_is2re
   :type: dict[str, str]

   

.. py:data:: S2EF_COUNTS

   

.. py:function:: get_data(datadir: str, task: str, split: str | None, del_intmd_files: bool) -> None


.. py:function:: uncompress_data(compressed_dir: str) -> str


.. py:function:: preprocess_data(uncompressed_dir: str, output_path: str) -> None


.. py:function:: verify_count(output_path: str, task: str, split: str) -> None


.. py:function:: cleanup(filename: str, dirname: str) -> None


.. py:data:: parser

   

