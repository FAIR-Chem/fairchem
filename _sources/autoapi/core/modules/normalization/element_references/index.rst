core.modules.normalization.element_references
=============================================

.. py:module:: core.modules.normalization.element_references

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.modules.normalization.element_references.LinearReferences


Functions
---------

.. autoapisummary::

   core.modules.normalization.element_references.create_element_references
   core.modules.normalization.element_references.fit_linear_references
   core.modules.normalization.element_references.load_references_from_config


Module Contents
---------------

.. py:class:: LinearReferences(element_references: torch.Tensor | None = None, max_num_elements: int = 118)

   Bases: :py:obj:`torch.nn.Module`


   Represents an elemental linear references model for a target property.

   In an elemental reference associates a value with each chemical element present in the dataset.
   Elemental references define a chemical composition model, i.e. a rough approximation of a target
   property (energy) using elemental references is done by summing the elemental references multiplied
   by the number of times the corresponding element is present.

   Elemental references energies can be taken as:
    - the energy of a chemical species in its elemental state
      (i.e. lowest energy polymorph of single element crystal structures for solids)
    - fitting a linear model to a dataset, where the features are the counts of each element in each data point.
      see the function fit_linear references below for details

   Training GNNs to predict the difference between DFT and the predictions of a chemical composition
   model represent a useful normalization scheme that can improve model accuracy. See for example the
   "Alternative reference scheme" section of the OC22 manuscript: https://arxiv.org/pdf/2206.08917


   .. py:method:: _apply_refs(target: torch.Tensor, batch: torch_geometric.data.Batch, sign: int, reshaped: bool = True) -> torch.Tensor

      Apply references batch-wise



   .. py:method:: dereference(target: torch.Tensor, batch: torch_geometric.data.Batch, reshaped: bool = True) -> torch.Tensor

      Remove linear references



   .. py:method:: forward(target: torch.Tensor, batch: torch_geometric.data.Batch, reshaped: bool = True) -> torch.Tensor

      Add linear references



.. py:function:: create_element_references(file: str | pathlib.Path | None = None, state_dict: dict | None = None) -> LinearReferences

   Create an element reference module.

   :param type: type of reference (only linear implemented)
   :type type: str
   :param file: path to pt or npz file
   :type file: str or Path
   :param state_dict: a state dict of a element reference module
   :type state_dict: dict

   :returns: LinearReference


.. py:function:: fit_linear_references(targets: list[str], dataset: torch.utils.data.Dataset, batch_size: int, num_batches: int | None = None, num_workers: int = 0, max_num_elements: int = 118, log_metrics: bool = True, use_numpy: bool = True, driver: str | None = None, shuffle: bool = True, seed: int = 0) -> dict[str, LinearReferences]

   Fit a set linear references for a list of targets using a given number of batches.

   :param targets: list of target names
   :param dataset: data set to fit linear references with
   :param batch_size: size of batch
   :param num_batches: number of batches to use in fit. If not given will use all batches
   :param num_workers: number of workers to use in data loader.
                       Note setting num_workers > 1 leads to finicky multiprocessing issues when using this function
                       in distributed mode. The issue has to do with pickling the functions in load_references_from_config
                       see function below...
   :param max_num_elements: max number of elements in dataset. If not given will use an ambitious value of 118
   :param log_metrics: if true will compute MAE, RMSE and R2 score of fit and log.
   :param use_numpy: use numpy.linalg.lstsq instead of torch. This tends to give better solutions.
   :param driver: backend used to solve linear system. See torch.linalg.lstsq docs. Ignored if use_numpy=True
   :param shuffle: whether to shuffle when loading the dataset
   :param seed: random seed used to shuffle the sampler if shuffle=True

   :returns: dict of fitted LinearReferences objects


.. py:function:: load_references_from_config(config: dict[str, Any], dataset: torch.utils.data.Dataset, seed: int = 0, checkpoint_dir: str | pathlib.Path | None = None) -> dict[str, LinearReferences]

   Create a dictionary with element references from a config.


