core.modules.normalization.normalizer
=====================================

.. py:module:: core.modules.normalization.normalizer

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.modules.normalization.normalizer.Normalizer


Functions
---------

.. autoapisummary::

   core.modules.normalization.normalizer.create_normalizer
   core.modules.normalization.normalizer.fit_normalizers
   core.modules.normalization.normalizer.load_normalizers_from_config


Module Contents
---------------

.. py:class:: Normalizer(mean: float | torch.Tensor = 0.0, rmsd: float | torch.Tensor = 1.0)

   Bases: :py:obj:`torch.nn.Module`


   Normalize/denormalize a tensor and optionally add a atom reference offset.


   .. py:method:: norm(tensor: torch.Tensor) -> torch.Tensor


   .. py:method:: denorm(normed_tensor: torch.Tensor) -> torch.Tensor


   .. py:method:: forward(normed_tensor: torch.Tensor) -> torch.Tensor


   .. py:method:: load_state_dict(state_dict: collections.abc.Mapping[str, Any], strict: bool = True, assign: bool = False)

      Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

      If :attr:`strict` is ``True``, then
      the keys of :attr:`state_dict` must exactly match the keys returned
      by this module's :meth:`~torch.nn.Module.state_dict` function.

      .. warning::
          If :attr:`assign` is ``True`` the optimizer must be created after
          the call to :attr:`load_state_dict`.

      :param state_dict: a dict containing parameters and
                         persistent buffers.
      :type state_dict: dict
      :param strict: whether to strictly enforce that the keys
                     in :attr:`state_dict` match the keys returned by this module's
                     :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
      :type strict: bool, optional
      :param assign: whether to assign items in the state
                     dictionary to their corresponding keys in the module instead
                     of copying them inplace into the module's current parameters and buffers.
                     When ``False``, the properties of the tensors in the current
                     module are preserved while when ``True``, the properties of the
                     Tensors in the state dict are preserved.
                     Default: ``False``
      :type assign: bool, optional

      :returns:     * **missing_keys** is a list of str containing the missing keys
                    * **unexpected_keys** is a list of str containing the unexpected keys
      :rtype: ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields

      .. note::

         If a parameter or buffer is registered as ``None`` and its corresponding key
         exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
         ``RuntimeError``.



.. py:function:: create_normalizer(file: str | pathlib.Path | None = None, state_dict: dict | None = None, tensor: torch.Tensor | None = None, mean: float | torch.Tensor | None = None, rmsd: float | torch.Tensor | None = None, stdev: float | torch.Tensor | None = None) -> Normalizer

   Build a target data normalizers with optional atom ref

   Only one of file, state_dict, tensor, or (mean and rmsd) will be used to create a normalizer.
   If more than one set of inputs are given priority will be given following the order in which they are listed above.

   :param file: path to pt or npz file.
   :type file: str or Path
   :param state_dict: a state dict for Normalizer module
   :type state_dict: dict
   :param tensor: a tensor with target values used to compute mean and std
   :type tensor: Tensor
   :param mean: mean of target data
   :type mean: float | Tensor
   :param rmsd: rmsd of target data, rmsd from mean = stdev, rmsd from 0 = rms
   :type rmsd: float | Tensor
   :param stdev: standard deviation (deprecated, use rmsd instead)

   :returns: Normalizer


.. py:function:: fit_normalizers(targets: list[str], dataset: torch.utils.data.Dataset, batch_size: int, override_values: dict[str, dict[str, float]] | None = None, rmsd_correction: int | None = None, element_references: dict | None = None, num_batches: int | None = None, num_workers: int = 0, shuffle: bool = True, seed: int = 0) -> dict[str, Normalizer]

   Estimate mean and rmsd from data to create normalizers

   :param targets: list of target names
   :param dataset: data set to fit linear references with
   :param batch_size: size of batch
   :param override_values: dictionary with target names and values to override. i.e. {"forces": {"mean": 0.0}} will set
                           the forces mean to zero.
   :param rmsd_correction: correction to use when computing mean in std/rmsd. See docs for torch.std.
                           If not given, will always use 0 when mean == 0, and 1 otherwise.
   :param element_references:
   :param num_batches: number of batches to use in fit. If not given will use all batches
   :param num_workers: number of workers to use in data loader
                       Note setting num_workers > 1 leads to finicky multiprocessing issues when using this function
                       in distributed mode. The issue has to do with pickling the functions in load_normalizers_from_config
                       see function below...
   :param shuffle: whether to shuffle when loading the dataset
   :param seed: random seed used to shuffle the sampler if shuffle=True

   :returns: dict of normalizer objects


.. py:function:: load_normalizers_from_config(config: dict[str, Any], dataset: torch.utils.data.Dataset, seed: int = 0, checkpoint_dir: str | pathlib.Path | None = None, element_references: dict[str, fairchem.core.modules.normalization.element_references.LinearReferences] | None = None) -> dict[str, Normalizer]

   Create a dictionary with element references from a config.


