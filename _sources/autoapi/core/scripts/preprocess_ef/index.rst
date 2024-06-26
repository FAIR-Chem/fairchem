core.scripts.preprocess_ef
==========================

.. py:module:: core.scripts.preprocess_ef

.. autoapi-nested-parse::

   Creates LMDB files with extracted graph features from provided *.extxyz files
   for the S2EF task.



Attributes
----------

.. autoapisummary::

   core.scripts.preprocess_ef.parser


Functions
---------

.. autoapisummary::

   core.scripts.preprocess_ef.write_images_to_lmdb
   core.scripts.preprocess_ef.main
   core.scripts.preprocess_ef.get_parser


Module Contents
---------------

.. py:function:: write_images_to_lmdb(mp_arg)

.. py:function:: main(args: argparse.Namespace) -> None

.. py:function:: get_parser() -> argparse.ArgumentParser

.. py:data:: parser
   :type:  argparse.ArgumentParser

