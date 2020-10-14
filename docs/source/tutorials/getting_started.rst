Getting Started
===============

Installation
************

The easiest way to install the prerequisites is via `conda <https://docs.conda.io/en/latest/>`_.
After installing conda, run the following commands to create a new environment named
ocp-models and install dependencies:

Pre-install step
----------------

Install conda-merge:

.. code-block:: sh

    pip install conda-merge

If you're using system pip, then you may want to add the --user flag to avoid using sudo.
Check that you can invoke conda-merge by running conda-merge -h.

GPU machines
------------

Instructions are for PyTorch 1.6, CUDA 10.1 specifically.

First, check that CUDA is in your PATH and LD_LIBRARY_PATH, e.g.

.. code-block:: sh

    $ echo $PATH | tr ':' '\n' | grep cuda
    /public/apps/cuda/10.1/bin
    $ echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
    /public/apps/cuda/10.1/lib64

The exact paths may differ on your system. Then install the dependencies:

.. code-block:: sh

    conda-merge env.common.yml env.gpu.yml > env.yml
    conda env create -f env.yml

Activate the conda environment with conda activate ocp-models. Install this package with pip install -e .. Finally, install the pre-commit hooks:

.. code-block:: sh

    pre-commit install

CPU-only machines
-----------------

Please skip the following if you completed the with-GPU installation from above.

.. code-block:: sh

    conda-merge env.common.yml env.cpu.yml > env.yml
    conda env create -f env.yml
    conda activate ocp-models
    pip install -e .
    pre-commit install

Activate the conda environment with conda activate ocp-models. Install this package with pip install -e .. Finally, install the pre-commit hooks:

.. code-block:: sh

    pre-commit install


Dataset
*******

Dataset download links can be found at `Open Catalysis Project website <http://www.opencatalstproject.org>`_
for the S2EF, IS2RS, and IS2RE tasks. IS2* datasets are stored as LMDB files and are ready
to be used upon download. S2EF datasets require an additional preprocessing step.

Preprocess datasets - S2EF only
-------------------------------

For the S2EF task, run:

1. Download the dataset of interest: :code:`curl -O download_link`
2. Untar the downloaded dataset: :code:`tar -xzvf dataset_name.tar`
3. Uncompress the untarred directory contents: :code:`python ocp/scripts/uncompress.py --ipdir /path/to/dataset_name_compressed --opdir raw_data/`
4. Run the LMDB preprocessing script:
:code:`scripts/preprocess_ef.py --data-path raw_data/ --out-path processed_lmdb/ \ `
:code:`--num-workers 32 --get-edges --ref-energy`; where
    - :code:`--get-edges`: includes edge information in LMDBs (~10x storage requirement, ~3-5x slowdown), otherwise, compute edges on the fly (larger GPU memory requirement).
    - :code:`--ref-energy`: uses referenced energies instead of raw energies.
