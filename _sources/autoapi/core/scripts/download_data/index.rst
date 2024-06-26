core.scripts.download_data
==========================

.. py:module:: core.scripts.download_data


Attributes
----------

.. autoapisummary::

   core.scripts.download_data.DOWNLOAD_LINKS_s2ef
   core.scripts.download_data.DOWNLOAD_LINKS_is2re
   core.scripts.download_data.S2EF_COUNTS
   core.scripts.download_data.parser


Functions
---------

.. autoapisummary::

   core.scripts.download_data.get_data
   core.scripts.download_data.uncompress_data
   core.scripts.download_data.preprocess_data
   core.scripts.download_data.verify_count
   core.scripts.download_data.cleanup


Module Contents
---------------

.. py:data:: DOWNLOAD_LINKS_s2ef
   :type:  dict[str, dict[str, str]]

.. py:data:: DOWNLOAD_LINKS_is2re
   :type:  dict[str, str]

.. py:data:: S2EF_COUNTS

.. py:function:: get_data(datadir: str, task: str, split: str | None, del_intmd_files: bool) -> None

.. py:function:: uncompress_data(compressed_dir: str) -> str

.. py:function:: preprocess_data(uncompressed_dir: str, output_path: str) -> None

.. py:function:: verify_count(output_path: str, task: str, split: str) -> None

.. py:function:: cleanup(filename: str, dirname: str) -> None

.. py:data:: parser

