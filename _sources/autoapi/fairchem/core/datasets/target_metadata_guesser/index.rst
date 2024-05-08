:py:mod:`fairchem.core.datasets.target_metadata_guesser`
========================================================

.. py:module:: fairchem.core.datasets.target_metadata_guesser


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.core.datasets.target_metadata_guesser.uniform_atoms_lengths
   fairchem.core.datasets.target_metadata_guesser.target_constant_shape
   fairchem.core.datasets.target_metadata_guesser.target_per_atom
   fairchem.core.datasets.target_metadata_guesser.target_extensive
   fairchem.core.datasets.target_metadata_guesser.guess_target_metadata
   fairchem.core.datasets.target_metadata_guesser.guess_property_metadata



.. py:function:: uniform_atoms_lengths(atoms_lens) -> bool


.. py:function:: target_constant_shape(atoms_lens, target_samples) -> bool


.. py:function:: target_per_atom(atoms_lens, target_samples) -> bool


.. py:function:: target_extensive(atoms_lens, target_samples, threshold: float = 0.2)


.. py:function:: guess_target_metadata(atoms_len, target_samples)


.. py:function:: guess_property_metadata(atoms_list)


