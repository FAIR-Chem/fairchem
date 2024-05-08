:py:mod:`fairchem.core.scripts.uncompress`
==========================================

.. py:module:: fairchem.core.scripts.uncompress

.. autoapi-nested-parse::

   Uncompresses downloaded S2EF datasets to be used by the LMDB preprocessing
   script - preprocess_ef.py



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.scripts.uncompress.read_lzma
   fairchem.core.scripts.uncompress.decompress_list_of_files
   fairchem.core.scripts.uncompress.get_parser
   fairchem.core.scripts.uncompress.main



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.core.scripts.uncompress.parser


.. py:function:: read_lzma(inpfile: str, outfile: str) -> None


.. py:function:: decompress_list_of_files(ip_op_pair: tuple[str, str]) -> None


.. py:function:: get_parser() -> argparse.ArgumentParser


.. py:function:: main(args: argparse.Namespace) -> None


.. py:data:: parser
   :type: argparse.ArgumentParser

   

