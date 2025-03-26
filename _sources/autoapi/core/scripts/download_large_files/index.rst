core.scripts.download_large_files
=================================

.. py:module:: core.scripts.download_large_files


Attributes
----------

.. autoapisummary::

   core.scripts.download_large_files.S3_ROOT
   core.scripts.download_large_files.FILE_GROUPS
   core.scripts.download_large_files.args


Functions
---------

.. autoapisummary::

   core.scripts.download_large_files.parse_args
   core.scripts.download_large_files.change_path_for_pypi
   core.scripts.download_large_files.download_file_group


Module Contents
---------------

.. py:data:: S3_ROOT
   :value: 'https://dl.fbaipublicfiles.com/opencatalystproject/data/large_files/'


.. py:data:: FILE_GROUPS

.. py:function:: parse_args()

.. py:function:: change_path_for_pypi(files_to_download: list[pathlib.Path], par_dir: str, install_dir: pathlib.Path, test_par_dir: pathlib.Path | None) -> list[pathlib.Path]

   Modify or exclude files from download if running in a PyPi-installed
   build.

   Installation of FAIR-Chem with PyPi does not include the entire
   directory structure of the fairchem repo. As such, files outside of
   `src` can't be downloaded and those in `src` should actually be in the
   `site-packages` directory. If a user wants these files, they must
   build from the git repo.

   If the tests have been separately downloaded (e.g. from the git repo),
   then we can download if we've been told where those tests have been
   downloaded to. Note that we can't divine that location from anything
   in fairchem.core because they would have to be somewhere "unexpected"
   since we've built with PyPi which shouldn't have tests at all.

   :param files_to_download: List of files to be downloaded
   :param par_dir: the parent directory of the PyPi build,
                   probably "site-packages"
   :param install_dir: path to where fairchem.core was installed
   :param test_par_dir: path to where tests have been downloaded
                        (not necessarily the same as install_dir)
   :return: modified list of files to be downloaded


.. py:function:: download_file_group(file_group: str, test_par_dir: pathlib.Path | None = None) -> None

   Download the given file group.

   :param file_group: Name of group of files to download
   :param test_par_dir: Parent directory where fairchem tests have been
                        downloaded


.. py:data:: args

