:py:mod:`ocpmodels.common.utils`
================================

.. py:module:: ocpmodels.common.utils

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.common.utils.Complete
   ocpmodels.common.utils.SeverityLevelBetween



Functions
~~~~~~~~~

.. autoapisummary::

   ocpmodels.common.utils.pyg2_data_transform
   ocpmodels.common.utils.save_checkpoint
   ocpmodels.common.utils.warmup_lr_lambda
   ocpmodels.common.utils.print_cuda_usage
   ocpmodels.common.utils.conditional_grad
   ocpmodels.common.utils.plot_histogram
   ocpmodels.common.utils.collate
   ocpmodels.common.utils.add_edge_distance_to_graph
   ocpmodels.common.utils._import_local_file
   ocpmodels.common.utils.setup_experimental_imports
   ocpmodels.common.utils._get_project_root
   ocpmodels.common.utils.setup_imports
   ocpmodels.common.utils.dict_set_recursively
   ocpmodels.common.utils.parse_value
   ocpmodels.common.utils.create_dict_from_args
   ocpmodels.common.utils.load_config
   ocpmodels.common.utils.build_config
   ocpmodels.common.utils.create_grid
   ocpmodels.common.utils.save_experiment_log
   ocpmodels.common.utils.get_pbc_distances
   ocpmodels.common.utils.radius_graph_pbc
   ocpmodels.common.utils.get_max_neighbors_mask
   ocpmodels.common.utils.get_pruned_edge_idx
   ocpmodels.common.utils.merge_dicts
   ocpmodels.common.utils.setup_logging
   ocpmodels.common.utils.compute_neighbors
   ocpmodels.common.utils.check_traj_files
   ocpmodels.common.utils.new_trainer_context
   ocpmodels.common.utils._resolve_scale_factor_submodule
   ocpmodels.common.utils._report_incompat_keys
   ocpmodels.common.utils.load_state_dict
   ocpmodels.common.utils.scatter_det
   ocpmodels.common.utils.get_commit_hash
   ocpmodels.common.utils.cg_change_mat
   ocpmodels.common.utils.irreps_sum
   ocpmodels.common.utils.update_config
   ocpmodels.common.utils.get_loss_module



.. py:function:: pyg2_data_transform(data: torch_geometric.data.Data)

   if we're on the new pyg (2.0 or later) and if the Data stored is in older format
   we need to convert the data to the new format


.. py:function:: save_checkpoint(state, checkpoint_dir: str = 'checkpoints/', checkpoint_file: str = 'checkpoint.pt') -> str


.. py:class:: Complete


   .. py:method:: __call__(data)



.. py:function:: warmup_lr_lambda(current_step: int, optim_config)

   Returns a learning rate multiplier.
   Till `warmup_steps`, learning rate linearly increases to `initial_lr`,
   and then gets multiplied by `lr_gamma` every time a milestone is crossed.


.. py:function:: print_cuda_usage() -> None


.. py:function:: conditional_grad(dec)

   Decorator to enable/disable grad depending on whether force/energy predictions are being made


.. py:function:: plot_histogram(data, xlabel: str = '', ylabel: str = '', title: str = '')


.. py:function:: collate(data_list)


.. py:function:: add_edge_distance_to_graph(batch, device='cpu', dmin: float = 0.0, dmax: float = 6.0, num_gaussians: int = 50)


.. py:function:: _import_local_file(path: pathlib.Path, *, project_root: pathlib.Path) -> None

   Imports a Python file as a module

   :param path: The path to the file to import
   :type path: Path
   :param project_root: The root directory of the project (i.e., the "ocp" folder)
   :type project_root: Path


.. py:function:: setup_experimental_imports(project_root: pathlib.Path) -> None

   Import selected directories of modules from the "experimental" subdirectory.

   If a file named ".include" is present in the "experimental" subdirectory,
   this will be read as a list of experimental subdirectories whose module
   (including in any subsubdirectories) should be imported.

   :param project_root: The root directory of the project (i.e., the "ocp" folder)


.. py:function:: _get_project_root() -> pathlib.Path

   Gets the root folder of the project (the "ocp" folder)
   :return: The absolute path to the project root.


.. py:function:: setup_imports(config: dict | None = None) -> None


.. py:function:: dict_set_recursively(dictionary, key_sequence, val) -> None


.. py:function:: parse_value(value)

   Parse string as Python literal if possible and fallback to string.


.. py:function:: create_dict_from_args(args: list, sep: str = '.')

   Create a (nested) dictionary from console arguments.
   Keys in different dictionary levels are separated by sep.


.. py:function:: load_config(path: str, previous_includes: list | None = None)


.. py:function:: build_config(args, args_override)


.. py:function:: create_grid(base_config, sweep_file: str)


.. py:function:: save_experiment_log(args, jobs, configs)


.. py:function:: get_pbc_distances(pos, edge_index, cell, cell_offsets, neighbors, return_offsets: bool = False, return_distance_vec: bool = False)


.. py:function:: radius_graph_pbc(data, radius, max_num_neighbors_threshold, enforce_max_neighbors_strictly: bool = False, pbc=None)


.. py:function:: get_max_neighbors_mask(natoms, index, atom_distance, max_num_neighbors_threshold, degeneracy_tolerance: float = 0.01, enforce_max_strictly: bool = False)

   Give a mask that filters out edges so that each atom has at most
   `max_num_neighbors_threshold` neighbors.
   Assumes that `index` is sorted.

   Enforcing the max strictly can force the arbitrary choice between
   degenerate edges. This can lead to undesired behaviors; for
   example, bulk formation energies which are not invariant to
   unit cell choice.

   A degeneracy tolerance can help prevent sudden changes in edge
   existence from small changes in atom position, for example,
   rounding errors, slab relaxation, temperature, etc.


.. py:function:: get_pruned_edge_idx(edge_index, num_atoms: int, max_neigh: float = 1000000000.0) -> torch.Tensor


.. py:function:: merge_dicts(dict1: dict, dict2: dict)

   Recursively merge two dictionaries.
   Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
   value, this will call itself recursively to merge these dictionaries.
   This does not modify the input dictionaries (creates an internal copy).
   Additionally returns a list of detected duplicates.
   Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py

   :param dict1: First dict.
   :type dict1: dict
   :param dict2: Second dict. Values in dict2 will override values from dict1 in case they share the same key.
   :type dict2: dict

   :returns: **return_dict** -- Merged dictionaries.
   :rtype: dict


.. py:class:: SeverityLevelBetween(min_level: int, max_level: int)


   Bases: :py:obj:`logging.Filter`

   Filter instances are used to perform arbitrary filtering of LogRecords.

   Loggers and Handlers can optionally use Filter instances to filter
   records as desired. The base filter class only allows events which are
   below a certain point in the logger hierarchy. For example, a filter
   initialized with "A.B" will allow events logged by loggers "A.B",
   "A.B.C", "A.B.C.D", "A.B.D" etc. but not "A.BB", "B.A.B" etc. If
   initialized with the empty string, all events are passed.

   .. py:method:: filter(record) -> bool

      Determine if the specified record is to be logged.

      Returns True if the record should be logged, or False otherwise.
      If deemed appropriate, the record may be modified in-place.



.. py:function:: setup_logging() -> None


.. py:function:: compute_neighbors(data, edge_index)


.. py:function:: check_traj_files(batch, traj_dir) -> bool


.. py:function:: new_trainer_context(*, config: dict[str, Any], args: argparse.Namespace)


.. py:function:: _resolve_scale_factor_submodule(model: torch.nn.Module, name: str)


.. py:function:: _report_incompat_keys(model: torch.nn.Module, keys: torch.nn.modules.module._IncompatibleKeys, strict: bool = False) -> tuple[list[str], list[str]]


.. py:function:: load_state_dict(module: torch.nn.Module, state_dict: collections.abc.Mapping[str, torch.Tensor], strict: bool = True) -> tuple[list[str], list[str]]


.. py:function:: scatter_det(*args, **kwargs)


.. py:function:: get_commit_hash()


.. py:function:: cg_change_mat(ang_mom: int, device: str = 'cpu') -> torch.tensor


.. py:function:: irreps_sum(ang_mom: int) -> int

   Returns the sum of the dimensions of the irreps up to the specified angular momentum.

   :param ang_mom: max angular momenttum to sum up dimensions of irreps


.. py:function:: update_config(base_config)

   Configs created prior to OCP 2.0 are organized a little different than they
   are now. Update old configs to fit the new expected structure.


.. py:function:: get_loss_module(loss_name)


