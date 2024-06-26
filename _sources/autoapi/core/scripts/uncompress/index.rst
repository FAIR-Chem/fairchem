core.scripts.uncompress
=======================

.. py:module:: core.scripts.uncompress

.. autoapi-nested-parse::

   Uncompresses downloaded S2EF datasets to be used by the LMDB preprocessing
   script - preprocess_ef.py



Attributes
----------

.. autoapisummary::

   core.scripts.uncompress.parser


Functions
---------

.. autoapisummary::

   core.scripts.uncompress.read_lzma
   core.scripts.uncompress.decompress_list_of_files
   core.scripts.uncompress.get_parser
   core.scripts.uncompress.main


Module Contents
---------------

.. py:function:: read_lzma(inpfile: str, outfile: str) -> None

.. py:function:: decompress_list_of_files(ip_op_pair: tuple[str, str]) -> None

.. py:function:: get_parser() -> argparse.ArgumentParser

.. py:function:: main(args: argparse.Namespace) -> None

.. py:data:: parser
   :type:  argparse.ArgumentParser

