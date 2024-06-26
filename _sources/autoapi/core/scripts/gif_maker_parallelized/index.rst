core.scripts.gif_maker_parallelized
===================================

.. py:module:: core.scripts.gif_maker_parallelized

.. autoapi-nested-parse::

   Script to generate gifs from traj

   Note:
   This is just a quick way to generate gifs and visalizations from traj, there are many parameters and settings in the code that people can vary to make visualizations better. We have chosen these settings as this seem to work fine for most of our systems.

   Requirements:

   povray
   ffmpeg
   ase==3.21



Attributes
----------

.. autoapisummary::

   core.scripts.gif_maker_parallelized.parser


Functions
---------

.. autoapisummary::

   core.scripts.gif_maker_parallelized.pov_from_atoms
   core.scripts.gif_maker_parallelized.parallelize_generation
   core.scripts.gif_maker_parallelized.get_parser


Module Contents
---------------

.. py:function:: pov_from_atoms(mp_args) -> None

.. py:function:: parallelize_generation(traj_path, out_path: str, n_procs) -> None

.. py:function:: get_parser() -> argparse.ArgumentParser

.. py:data:: parser
   :type:  argparse.ArgumentParser

