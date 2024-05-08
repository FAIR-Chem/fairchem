:py:mod:`fairchem.demo.ocpapi.workflows.adsorbates`
===================================================

.. py:module:: fairchem.demo.ocpapi.workflows.adsorbates


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   fairchem.demo.ocpapi.workflows.adsorbates.Lifetime
   fairchem.demo.ocpapi.workflows.adsorbates.AdsorbateSlabRelaxations
   fairchem.demo.ocpapi.workflows.adsorbates.AdsorbateBindingSites



Functions
~~~~~~~~~

.. autoapisummary::

   fairchem.demo.ocpapi.workflows.adsorbates._setup_log_record_factory
   fairchem.demo.ocpapi.workflows.adsorbates._ensure_model_supported
   fairchem.demo.ocpapi.workflows.adsorbates._get_bulk_if_supported
   fairchem.demo.ocpapi.workflows.adsorbates._ensure_adsorbate_supported
   fairchem.demo.ocpapi.workflows.adsorbates._get_slabs
   fairchem.demo.ocpapi.workflows.adsorbates._get_absorbate_configs_on_slab
   fairchem.demo.ocpapi.workflows.adsorbates._get_absorbate_configs_on_slab_with_logging
   fairchem.demo.ocpapi.workflows.adsorbates._get_adsorbate_configs_on_slabs
   fairchem.demo.ocpapi.workflows.adsorbates._submit_relaxations
   fairchem.demo.ocpapi.workflows.adsorbates._submit_relaxations_with_progress_logging
   fairchem.demo.ocpapi.workflows.adsorbates.get_adsorbate_slab_relaxation_results
   fairchem.demo.ocpapi.workflows.adsorbates.wait_for_adsorbate_slab_relaxations
   fairchem.demo.ocpapi.workflows.adsorbates._delete_system
   fairchem.demo.ocpapi.workflows.adsorbates._ensure_system_deleted
   fairchem.demo.ocpapi.workflows.adsorbates._run_relaxations_on_slab
   fairchem.demo.ocpapi.workflows.adsorbates._refresh_pbar
   fairchem.demo.ocpapi.workflows.adsorbates._relax_binding_sites_on_slabs
   fairchem.demo.ocpapi.workflows.adsorbates.find_adsorbate_binding_sites



Attributes
~~~~~~~~~~

.. autoapisummary::

   fairchem.demo.ocpapi.workflows.adsorbates._CTX_AD_BULK
   fairchem.demo.ocpapi.workflows.adsorbates._CTX_SLAB
   fairchem.demo.ocpapi.workflows.adsorbates.DEFAULT_CLIENT
   fairchem.demo.ocpapi.workflows.adsorbates._DEFAULT_ADSLAB_FILTER


.. py:data:: _CTX_AD_BULK
   :type: contextvars.ContextVar[Tuple[str, str]]

   

.. py:data:: _CTX_SLAB
   :type: contextvars.ContextVar[fairchem.demo.ocpapi.client.Slab]

   

.. py:function:: _setup_log_record_factory() -> None

   Adds a log record factory that stores information about the currently
   running job on a log message.


.. py:data:: DEFAULT_CLIENT
   :type: fairchem.demo.ocpapi.client.Client

   

.. py:exception:: AdsorbatesException


   Bases: :py:obj:`Exception`

   Base exception for all others in this module.


.. py:exception:: UnsupportedModelException(model: str, allowed_models: List[str])


   Bases: :py:obj:`AdsorbatesException`

   Exception raised when a model is not supported in the API.


.. py:exception:: UnsupportedBulkException(bulk: str)


   Bases: :py:obj:`AdsorbatesException`

   Exception raised when a bulk material is not supported in the API.


.. py:exception:: UnsupportedAdsorbateException(adsorbate: str)


   Bases: :py:obj:`AdsorbatesException`

   Exception raised when an adsorbate is not supported in the API.


.. py:class:: Lifetime(*args, **kwds)


   Bases: :py:obj:`enum.Enum`

   Represents different lifetimes when running relaxations.

   .. py:attribute:: SAVE

      The relaxation will be available on API servers indefinitely. It will not
      be possible to delete the relaxation in the future.

   .. py:attribute:: MARK_EPHEMERAL

      The relaxation will be saved on API servers, but can be deleted at any time
      in the future.

   .. py:attribute:: DELETE

      The relaxation will be deleted from API servers as soon as the results have
      been fetched.


.. py:class:: AdsorbateSlabRelaxations


   Stores the relaxations of adsorbate placements on the surface of a slab.

   .. py:attribute:: slab
      :type: fairchem.demo.ocpapi.client.Slab

      The slab on which the adsorbate was placed.

   .. py:attribute:: configs
      :type: List[fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationResult]

      Details of the relaxation of each adsorbate placement, including the
      final position.

   .. py:attribute:: system_id
      :type: str

      The ID of the system that stores all of the relaxations.

   .. py:attribute:: api_host
      :type: str

      The API host on which the relaxations were run.

   .. py:attribute:: ui_url
      :type: Optional[str]

      The URL at which results can be visualized.


.. py:class:: AdsorbateBindingSites


   Stores the inputs and results of a set of relaxations of adsorbate
   placements on the surface of a slab.

   .. py:attribute:: adsorbate
      :type: str

      Description of the adsorbate.

   .. py:attribute:: bulk
      :type: fairchem.demo.ocpapi.client.Bulk

      The bulk material that was being modeled.

   .. py:attribute:: model
      :type: str

      The type of the model that was run.

   .. py:attribute:: slabs
      :type: List[AdsorbateSlabRelaxations]

      The list of slabs that were generated from the bulk structure. Each
      contains its own list of adsorbate placements.


.. py:function:: _ensure_model_supported(client: fairchem.demo.ocpapi.client.Client, model: str) -> None
   :async:

   Checks that the input model is supported in the API.

   :param client: The client to use when making requests to the API.
   :param model: The model to check.

   :raises UnsupportedModelException: If the model is not supported.


.. py:function:: _get_bulk_if_supported(client: fairchem.demo.ocpapi.client.Client, bulk: str) -> fairchem.demo.ocpapi.client.Bulk
   :async:

   Returns the object from the input bulk if it is supported in the API.

   :param client: The client to use when making requests to the API.
   :param bulk: The bulk to fetch.

   :raises UnsupportedBulkException: If the requested bulk is not supported.

   :returns: Bulk instance for the input type.


.. py:function:: _ensure_adsorbate_supported(client: fairchem.demo.ocpapi.client.Client, adsorbate: str) -> None
   :async:

   Checks that the input adsorbate is supported in the API.

   :param client: The client to use when making requests to the API.
   :param adsorbate: The adsorbate to check.

   :raises UnsupportedAdsorbateException: If the adsorbate is not supported.


.. py:function:: _get_slabs(client: fairchem.demo.ocpapi.client.Client, bulk: fairchem.demo.ocpapi.client.Bulk) -> List[fairchem.demo.ocpapi.client.Slab]
   :async:

   Enumerates surfaces for the input bulk material.

   :param client: The client to use when making requests to the API.
   :param bulk: The bulk material from which slabs will be generated.

   :returns: The list of slabs that were generated.


.. py:function:: _get_absorbate_configs_on_slab(client: fairchem.demo.ocpapi.client.Client, adsorbate: str, slab: fairchem.demo.ocpapi.client.Slab) -> fairchem.demo.ocpapi.client.AdsorbateSlabConfigs
   :async:

   Generate initial guesses at adsorbate binding sites on the input slab.

   :param client: The client to use when making API calls.
   :param adsorbate: Description of the adsorbate to place.
   :param slab: The slab on which the adsorbate should be placed.

   :returns: An updated slab instance that has had tags applied to it and a list
             of Atoms objects, each with the positions of the adsorbate atoms on
             one of the candidate binding sites.


.. py:function:: _get_absorbate_configs_on_slab_with_logging(client: fairchem.demo.ocpapi.client.Client, adsorbate: str, slab: fairchem.demo.ocpapi.client.Slab) -> fairchem.demo.ocpapi.client.AdsorbateSlabConfigs
   :async:

   Wrapper around _get_absorbate_configs_on_slab that adds logging.


.. py:function:: _get_adsorbate_configs_on_slabs(client: fairchem.demo.ocpapi.client.Client, adsorbate: str, slabs: List[fairchem.demo.ocpapi.client.Slab]) -> List[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]
   :async:

   Finds candidate adsorbate binding sites on each of the input slabs.

   :param client: The client to use when making API calls.
   :param adsorbate: Description of the adsorbate to place.
   :param slabs: The slabs on which the adsorbate should be placed.

   :returns: List of slabs and, for each, the positions of the adsorbate
             atoms in the potential binding site.


.. py:function:: _submit_relaxations(client: fairchem.demo.ocpapi.client.Client, adsorbate: str, adsorbate_configs: List[fairchem.demo.ocpapi.client.Atoms], bulk: fairchem.demo.ocpapi.client.Bulk, slab: fairchem.demo.ocpapi.client.Slab, model: str, ephemeral: bool) -> str
   :async:

   Start relaxations for each of the input adsorbate configurations on the
   input slab.

   :param client: The client to use when making API calls.
   :param adsorbate: Description of the adsorbate to place.
   :param adsorbate_configs: Positions of the adsorbate on the slab. Each
                             will be relaxed independently.
   :param bulk: The bulk material from which the slab was generated.
   :param slab: The slab that should be searched for adsorbate binding sites.
   :param model: The model to use when evaluating forces and energies.
   :param ephemeral: Whether the relaxations should be marked as ephemeral.

   :returns: The system ID of the relaxation run, which can be used to fetch results
             as they become available.


.. py:function:: _submit_relaxations_with_progress_logging(client: fairchem.demo.ocpapi.client.Client, adsorbate: str, adsorbate_configs: List[fairchem.demo.ocpapi.client.Atoms], bulk: fairchem.demo.ocpapi.client.Bulk, slab: fairchem.demo.ocpapi.client.Slab, model: str, ephemeral: bool) -> str
   :async:

   Wrapper around _submit_relaxations that adds periodic logging in case
   calls to submit relaxations are being rate limited.


.. py:function:: get_adsorbate_slab_relaxation_results(system_id: str, config_ids: Optional[List[int]] = None, fields: Optional[List[str]] = None, client: fairchem.demo.ocpapi.client.Client = DEFAULT_CLIENT) -> List[fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationResult]
   :async:

   Wrapper around Client.get_adsorbate_slab_relaxations_results() that
   handles retries, including re-fetching individual configurations that
   are initially omitted.

   :param client: The client to use when making API calls.
   :param system_id: The system ID of the relaxations.
   :param config_ids: If defined and not empty, a subset of configurations
                      to fetch. Otherwise all configurations are returned.
   :param fields: If defined and not empty, a subset of fields in each
                  configuration to fetch. Otherwise all fields are returned.

   :returns: List of relaxation results, one for each adsorbate configuration in
             the system.


.. py:function:: wait_for_adsorbate_slab_relaxations(system_id: str, check_immediately: bool = False, slow_interval_sec: float = 30, fast_interval_sec: float = 10, pbar: Optional[tqdm.tqdm] = None, client: fairchem.demo.ocpapi.client.Client = DEFAULT_CLIENT) -> Dict[int, fairchem.demo.ocpapi.client.Status]
   :async:

   Blocks until all relaxations in the input system have finished, whether
   successfully or not.

   Relaxations are queued in the API, waiting until machines are ready to
   run them. Once started, they can take 1-2 minutes to finish. This method
   initially sleeps "slow_interval_sec" seconds between each check for any
   relaxations having finished. Once at least one result is ready, subsequent
   sleeps are for "fast_interval_sec" seconds.

   :param system_id: The ID of the system for which relaxations are running.
   :param check_immediately: If False (default), sleep before the first check
                             for relaxations having finished. If True, check whether relaxations
                             have finished immediately on entering this function.
   :param slow_interval_sec: The number of seconds to wait between each check
                             while all are still running.
   :param fast_interval_sec: The number of seconds to wait between each check
                             when at least one relaxation has finished in the system.
   :param pbar: A tqdm instance that tracks the number of configurations that
                have finished. This will be updated with the number of individual
                configurations whose relaxations have finished.
   :param client: The client to use when making API calls.

   :returns: Map of config IDs in the system to their terminal status.


.. py:function:: _delete_system(client: fairchem.demo.ocpapi.client.Client, system_id: str) -> None
   :async:

   Deletes the input system, with retries on failed attempts.

   :param client: The client to use when making API calls.
   :param system_id: The ID of the system to delete.


.. py:function:: _ensure_system_deleted(client: fairchem.demo.ocpapi.client.Client, system_id: str) -> AsyncGenerator[None, None]
   :async:

   Immediately yields control to the caller. When control returns to this
   function, try to delete the system with the input id.

   :param client: The client to use when making API calls.
   :param system_id: The ID of the system to delete.


.. py:function:: _run_relaxations_on_slab(client: fairchem.demo.ocpapi.client.Client, adsorbate: str, adsorbate_configs: List[fairchem.demo.ocpapi.client.Atoms], bulk: fairchem.demo.ocpapi.client.Bulk, slab: fairchem.demo.ocpapi.client.Slab, model: str, lifetime: Lifetime, pbar: tqdm.tqdm) -> AdsorbateSlabRelaxations
   :async:

   Start relaxations for each adsorbate configuration on the input slab
   and wait for all to finish.

   :param client: The client to use when making API calls.
   :param adsorbate: Description of the adsorbate to place.
   :param adsorbate_configs: The positions of atoms in each adsorbate placement
                             to be relaxed.
   :param bulk: The bulk material from which the slab was generated.
   :param slab: The slab that should be searched for adsorbate binding sites.
   :param model: The model to use when evaluating forces and energies.
   :param lifetime: Whether relaxations should be saved on the server, be marked
                    as ephemeral (allowing them to deleted in the future), or deleted
                    immediately.
   :param pbar: A progress bar to update as relaxations finish.

   :returns: Details of each adsorbate placement, including its relaxed position.


.. py:function:: _refresh_pbar(pbar: tqdm.tqdm, interval_sec: float) -> None
   :async:

   Helper function that refreshes the input progress bar on a regular
   schedule. This function never returns; it must be cancelled.

   :param pbar: The progress bar to refresh.
   :param interval_sec: The number of seconds to wait between each refresh.


.. py:function:: _relax_binding_sites_on_slabs(client: fairchem.demo.ocpapi.client.Client, adsorbate: str, bulk: fairchem.demo.ocpapi.client.Bulk, adslabs: List[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs], model: str, lifetime: Lifetime) -> AdsorbateBindingSites
   :async:

   Search for adsorbate binding sites on the input slab.

   :param client: The client to use when making API calls.
   :param adsorbate: Description of the adsorbate to place.
   :param bulk: The bulk material from which the slab was generated.
   :param adslabs: The slabs and, for each, the binding sites that should be
                   relaxed.
   :param model: The model to use when evaluating forces and energies.
   :param lifetime: Whether relaxations should be saved on the server, be marked
                    as ephemeral (allowing them to deleted in the future), or deleted
                    immediately.

   :returns: Details of each adsorbate placement, including its relaxed position.


.. py:data:: _DEFAULT_ADSLAB_FILTER
   :type: Callable[[List[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]], Awaitable[List[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]]]

   

.. py:function:: find_adsorbate_binding_sites(adsorbate: str, bulk: str, model: str = 'equiformer_v2_31M_s2ef_all_md', adslab_filter: Callable[[List[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]], Awaitable[List[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]]] = _DEFAULT_ADSLAB_FILTER, client: fairchem.demo.ocpapi.client.Client = DEFAULT_CLIENT, lifetime: Lifetime = Lifetime.SAVE) -> AdsorbateBindingSites
   :async:

   Search for adsorbate binding sites on surfaces of a bulk material.
   This executes the following steps:

       1. Ensure that both the adsorbate and bulk are supported in the
          OCP API.
       2. Enumerate unique surfaces from the bulk material.
       3. Enumerate likely binding sites for the input adsorbate on each
          of the generated surfaces.
       4. Filter the list of generated adsorbate/slab (adslab) configurations
           using the input adslab_filter.
       5. Relax each generated surface+adsorbate structure by refining
          atomic positions to minimize forces generated by the input model.

   :param adsorbate: Description of the adsorbate to place.
   :param bulk: The ID (typically Materials Project MP ID) of the bulk material
                on which the adsorbate will be placed.
   :param model: The type of the model to use when calculating forces during
                 relaxations.
   :param adslab_filter: A function that modifies the set of adsorbate/slab
                         configurations that will be relaxed. This can be used to subselect
                         slabs and/or adsorbate configurations.
   :param client: The OCP API client to use.
   :param lifetime: Whether relaxations should be saved on the server, be marked
                    as ephemeral (allowing them to deleted in the future), or deleted
                    immediately.

   :returns: Details of each adsorbate binding site, including results of relaxing
             to locally-optimized positions using the input model.

   :raises UnsupportedModelException: If the requested model is not supported.
   :raises UnsupportedBulkException: If the requested bulk is not supported.
   :raises UnsupportedAdsorbateException: If the requested adsorbate is not
       supported.


