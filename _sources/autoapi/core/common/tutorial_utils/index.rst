core.common.tutorial_utils
==========================

.. py:module:: core.common.tutorial_utils


Functions
---------

.. autoapisummary::

   core.common.tutorial_utils.fairchem_root
   core.common.tutorial_utils.fairchem_main
   core.common.tutorial_utils.describe_fairchem
   core.common.tutorial_utils.train_test_val_split
   core.common.tutorial_utils.generate_yml_config


Module Contents
---------------

.. py:function:: fairchem_root()

   Return the root directory of the installed fairchem-core package.


.. py:function:: fairchem_main()

   Return the path to fairchem main.py


.. py:function:: describe_fairchem()

   Print some system information that could be useful in debugging.


.. py:function:: train_test_val_split(ase_db, ttv=(0.8, 0.1, 0.1), files=('train.db', 'test.db', 'val.db'), seed=42)

   Split an ase db into train, test and validation dbs.

   ase_db: path to an ase db containing all the data.
   ttv: a tuple containing the fraction of train, test and val data. This will be normalized.
   files: a tuple of filenames to write the splits into. An exception is raised if these exist.
          You should delete them first.
   seed: an integer for the random number generator seed

   Returns the absolute path to files.


.. py:function:: generate_yml_config(checkpoint_path, yml='run.yml', delete=(), update=())

   Generate a yml config file from an existing checkpoint file.

   checkpoint_path: string to path of an existing checkpoint
   yml: name of file to write to.
   pop: list of keys to remove from the config
   update: dictionary of key:values to update

   Use a dot notation in update.

   Returns an absolute path to the generated yml file.


