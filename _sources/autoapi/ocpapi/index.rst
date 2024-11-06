ocpapi
======

.. py:module:: ocpapi

.. autoapi-nested-parse::

   Copyright (c) Meta, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/ocpapi/client/index
   /autoapi/ocpapi/version/index
   /autoapi/ocpapi/workflows/index


Attributes
----------

.. autoapisummary::

   ocpapi.__version__
   ocpapi.NO_LIMIT
   ocpapi.NoLimitType


Exceptions
----------

.. autoapisummary::

   ocpapi.NonRetryableRequestException
   ocpapi.RateLimitExceededException
   ocpapi.RequestException
   ocpapi.UnsupportedAdsorbateException
   ocpapi.UnsupportedBulkException
   ocpapi.UnsupportedModelException


Classes
-------

.. autoapisummary::

   ocpapi.Client
   ocpapi.Adsorbates
   ocpapi.AdsorbateSlabConfigs
   ocpapi.AdsorbateSlabRelaxationResult
   ocpapi.AdsorbateSlabRelaxationsRequest
   ocpapi.AdsorbateSlabRelaxationsResults
   ocpapi.AdsorbateSlabRelaxationsSystem
   ocpapi.Atoms
   ocpapi.Bulk
   ocpapi.Bulks
   ocpapi.Model
   ocpapi.Models
   ocpapi.Slab
   ocpapi.SlabMetadata
   ocpapi.Slabs
   ocpapi.Status
   ocpapi.AdsorbateBindingSites
   ocpapi.AdsorbateSlabRelaxations
   ocpapi.Lifetime
   ocpapi.keep_all_slabs
   ocpapi.keep_slabs_with_miller_indices
   ocpapi.prompt_for_slabs_to_keep
   ocpapi.RateLimitLogging


Functions
---------

.. autoapisummary::

   ocpapi.get_results_ui_url
   ocpapi.find_adsorbate_binding_sites
   ocpapi.get_adsorbate_slab_relaxation_results
   ocpapi.wait_for_adsorbate_slab_relaxations
   ocpapi.retry_api_calls


Package Contents
----------------

.. py:data:: __version__

.. py:class:: Client(host: str = 'open-catalyst-api.metademolab.com', scheme: str = 'https')

   Exposes each route in the OCP API as a method.


   .. py:attribute:: _host


   .. py:attribute:: _base_url


   .. py:property:: host
      :type: str


      The host being called by this client.


   .. py:method:: get_models() -> ocpapi.client.models.Models
      :async:


      Fetch the list of models that are supported in the API.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The models that are supported in the API.



   .. py:method:: get_bulks() -> ocpapi.client.models.Bulks
      :async:


      Fetch the list of bulk materials that are supported in the API.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The bulks that are supported throughout the API.



   .. py:method:: get_adsorbates() -> ocpapi.client.models.Adsorbates
      :async:


      Fetch the list of adsorbates that are supported in the API.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The adsorbates that are supported throughout the API.



   .. py:method:: get_slabs(bulk: str | ocpapi.client.models.Bulk) -> ocpapi.client.models.Slabs
      :async:


      Get a unique list of slabs for the input bulk structure.

      :param bulk: If a string, the id of the bulk to use. Otherwise the Bulk
                   instance to use.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: Slabs for each of the unique surfaces of the material.



   .. py:method:: get_adsorbate_slab_configs(adsorbate: str, slab: ocpapi.client.models.Slab) -> ocpapi.client.models.AdsorbateSlabConfigs
      :async:


      Get a list of possible binding sites for the input adsorbate on the
      input slab.

      :param adsorbate: Description of the the adsorbate to place.
      :param slab: Information about the slab on which the adsorbate should
                   be placed.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: Configurations for each adsorbate binding site on the slab.



   .. py:method:: submit_adsorbate_slab_relaxations(adsorbate: str, adsorbate_configs: list[ocpapi.client.models.Atoms], bulk: ocpapi.client.models.Bulk, slab: ocpapi.client.models.Slab, model: str, ephemeral: bool = False) -> ocpapi.client.models.AdsorbateSlabRelaxationsSystem
      :async:


      Starts relaxations of the input adsorbate configurations on the input
      slab using energies and forces returned by the input model. Relaxations
      are run asynchronously and results can be fetched using the system id
      that is returned from this method.

      :param adsorbate: Description of the adsorbate being simulated.
      :param adsorbate_configs: List of adsorbate configurations to relax. This
                                should only include the adsorbates themselves; the surface is
                                defined in the "slab" field that is a peer to this one.
      :param bulk: Details of the bulk material being simulated.
      :param slab: The structure of the slab on which adsorbates are placed.
      :param model: The model that will be used to evaluate energies and forces
                    during relaxations.
      :param ephemeral: If False (default), any later attempt to delete the
                        generated relaxations will be rejected. If True, deleting the
                        relaxations will be allowed, which is generally useful for
                        testing when there is no reason for results to be persisted.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: IDs of the relaxations.



   .. py:method:: get_adsorbate_slab_relaxations_request(system_id: str) -> ocpapi.client.models.AdsorbateSlabRelaxationsRequest
      :async:


      Fetches the original relaxations request for the input system.

      :param system_id: The ID of the system to fetch.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The original request that was made when submitting relaxations.



   .. py:method:: get_adsorbate_slab_relaxations_results(system_id: str, config_ids: list[int] | None = None, fields: list[str] | None = None) -> ocpapi.client.models.AdsorbateSlabRelaxationsResults
      :async:


      Fetches relaxation results for the input system.

      :param system_id: The system id of the relaxations.
      :param config_ids: If defined and not empty, a subset of configurations
                         to fetch. Otherwise all configurations are returned.
      :param fields: If defined and not empty, a subset of fields in each
                     configuration to fetch. Otherwise all fields are returned.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The relaxation results for each configuration in the system.



   .. py:method:: delete_adsorbate_slab_relaxations(system_id: str) -> None
      :async:


      Deletes all relaxation results for the input system.

      :param system_id: The ID of the system to delete.

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.



   .. py:method:: _run_request(path: str, method: str, **kwargs) -> str
      :async:


      Helper method that runs the input request on a thread so that
      it doesn't block the event loop on the calling thread.

      :param path: The URL path to make the request against.
      :param method: The HTTP method to use (GET, POST, etc.).

      :raises RateLimitExceededException: If the call was rejected because a
          server side rate limit was breached.
      :raises NonRetryableRequestException: If the call was rejected and a retry
          is not expected to succeed.
      :raises RequestException: For all other errors when making the request; it
          is possible, though not guaranteed, that a retry could succeed.

      :returns: The response body from the request as a string.



.. py:exception:: NonRetryableRequestException(method: str, url: str, cause: str)

   Bases: :py:obj:`RequestException`


   Exception raised when an API call is rejected for a reason that will
   not succeed on retry. For example, this might include a malformed request
   or action that is not allowed.


.. py:exception:: RateLimitExceededException(method: str, url: str, retry_after: datetime.timedelta | None = None)

   Bases: :py:obj:`RequestException`


   Exception raised when an API call is rejected because a rate limit has
   been exceeded.

   .. attribute:: retry_after

      If known, the time to wait before the next attempt to
      call the API should be made.


   .. py:attribute:: retry_after
      :type:  datetime.timedelta | None


.. py:exception:: RequestException(method: str, url: str, cause: str)

   Bases: :py:obj:`Exception`


   Exception raised any time there is an error while making an API call.


.. py:class:: Adsorbates

   Bases: :py:obj:`_DataModel`


   Stores the response from a request to fetch adsorbates supported in the
   API.


   .. py:attribute:: adsorbates_supported
      :type:  List[str]

      List of adsorbates that can be used in the API.


.. py:class:: AdsorbateSlabConfigs

   Bases: :py:obj:`_DataModel`


   Stores the response from a request to fetch placements of a single
   absorbate on a slab.


   .. py:attribute:: adsorbate_configs
      :type:  List[Atoms]

      List of structures, each representing one possible adsorbate placement.


   .. py:attribute:: slab
      :type:  Slab

      The structure of the slab on which the adsorbate is placed.


.. py:class:: AdsorbateSlabRelaxationResult

   Bases: :py:obj:`_DataModel`


   Stores information about a single adsorbate slab configuration, including
   outputs for the model used in relaxations.

   The API to fetch relaxation results supports requesting a subset of fields
   in order to limit the size of response payloads. Optional attributes will
   be defined only if they are including the response.


   .. py:attribute:: config_id
      :type:  int

      ID of the configuration within the system.


   .. py:attribute:: status
      :type:  Status

      The status of the request for information about this configuration.


   .. py:attribute:: system_id
      :type:  Optional[str]

      The ID of the system in which the configuration was originally submitted.


   .. py:attribute:: cell
      :type:  Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]

      3x3 matrix with unit cell vectors.


   .. py:attribute:: pbc
      :type:  Optional[Tuple[bool, bool, bool]]

      Whether the structure is periodic along the a, b, and c lattice vectors,
      respectively.


   .. py:attribute:: numbers
      :type:  Optional[List[int]]

      The atomic number of each atom in the unit cell.


   .. py:attribute:: positions
      :type:  Optional[List[Tuple[float, float, float]]]

      The coordinates of each atom in the unit cell, relative to the cartesian
      frame.


   .. py:attribute:: tags
      :type:  Optional[List[int]]

      Labels for each atom in the unit cell where 0 represents a subsurface atom
      (fixed during optimization), 1 represents a surface atom, and 2 represents
      an adsorbate atom.


   .. py:attribute:: energy
      :type:  Optional[float]

      The energy of the configuration.


   .. py:attribute:: energy_trajectory
      :type:  Optional[List[float]]

      The energy of the configuration at each point along the relaxation
      trajectory.


   .. py:attribute:: forces
      :type:  Optional[List[Tuple[float, float, float]]]

      The forces on each atom in the relaxed structure.


   .. py:method:: to_ase_atoms() -> ase.Atoms

      Creates an ase.Atoms object with the positions, element numbers,
      etc. populated from values on this object.

      The predicted energy and forces will also be copied to the new
      ase.Atoms object as a SinglePointCalculator (a calculator that
      stores the results of an already-run simulation).

      :returns: ase.Atoms object with values from this object.



.. py:class:: AdsorbateSlabRelaxationsRequest

   Bases: :py:obj:`_DataModel`


   Stores the request to submit a new batch of adsorbate slab relaxations.


   .. py:attribute:: adsorbate
      :type:  str

      Description of the adsorbate.


   .. py:attribute:: adsorbate_configs
      :type:  List[Atoms]

      List of adsorbate placements being relaxed.


   .. py:attribute:: bulk
      :type:  Bulk

      Information about the original bulk structure used to create the slab.


   .. py:attribute:: slab
      :type:  Slab

      The structure of the slab on which adsorbates are placed.


   .. py:attribute:: model
      :type:  str

      The type of the ML model being used during relaxations.


   .. py:attribute:: ephemeral
      :type:  Optional[bool]

      Whether the relaxations can be deleted (assume they cannot be deleted if
      None).


   .. py:attribute:: adsorbate_reaction
      :type:  Optional[str]

      If possible, an html-formatted string describing the reaction will be added
      to this field.


.. py:class:: AdsorbateSlabRelaxationsResults

   Bases: :py:obj:`_DataModel`


   Stores the response from a request for results of adsorbate slab
   relaxations.


   .. py:attribute:: configs
      :type:  List[AdsorbateSlabRelaxationResult]

      List of configurations in the system, each representing one placement of
      an adsorbate on a slab surface.


   .. py:attribute:: omitted_config_ids
      :type:  List[int]

      List of IDs of configurations that were requested but omitted by the
      server. Results for these IDs can be requested again.


.. py:class:: AdsorbateSlabRelaxationsSystem

   Bases: :py:obj:`_DataModel`


   Stores the response from a request to submit a new batch of adsorbate
   slab relaxations.


   .. py:attribute:: system_id
      :type:  str

      Unique ID for this set of relaxations which can be used to fetch results
      later.


   .. py:attribute:: config_ids
      :type:  List[int]

      The list of IDs assigned to each of the input adsorbate placements, in the
      same order in which they were submitted.


.. py:class:: Atoms

   Bases: :py:obj:`_DataModel`


   Subset of the fields from an ASE Atoms object that are used within this
   API.


   .. py:attribute:: cell
      :type:  Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]

      3x3 matrix with unit cell vectors.


   .. py:attribute:: pbc
      :type:  Tuple[bool, bool, bool]

      Whether the structure is periodic along the a, b, and c lattice vectors,
      respectively.


   .. py:attribute:: numbers
      :type:  List[int]

      The atomic number of each atom in the unit cell.


   .. py:attribute:: positions
      :type:  List[Tuple[float, float, float]]

      The coordinates of each atom in the unit cell, relative to the cartesian
      frame.


   .. py:attribute:: tags
      :type:  List[int]

      Labels for each atom in the unit cell where 0 represents a subsurface atom
      (fixed during optimization), 1 represents a surface atom, and 2 represents
      an adsorbate atom.


   .. py:method:: to_ase_atoms() -> ase.Atoms

      Creates an ase.Atoms object with the positions, element numbers,
      etc. populated from values on this object.

      :returns: ase.Atoms object with values from this object.



.. py:class:: Bulk

   Bases: :py:obj:`_DataModel`


   Stores information about a single bulk material.


   .. py:attribute:: src_id
      :type:  str

      The ID of the material.


   .. py:attribute:: formula
      :type:  str

      The chemical formula of the material.


   .. py:attribute:: elements
      :type:  List[str]

      The list of elements in the material.


.. py:class:: Bulks

   Bases: :py:obj:`_DataModel`


   Stores the response from a request to fetch bulks supported in the API.


   .. py:attribute:: bulks_supported
      :type:  List[Bulk]

      List of bulks that can be used in the API.


.. py:class:: Model

   Bases: :py:obj:`_DataModel`


   Stores information about a single model supported in the API.


   .. py:attribute:: id
      :type:  str

      The ID of the model.


.. py:class:: Models

   Bases: :py:obj:`_DataModel`


   Stores the response from a request for models supported in the API.


   .. py:attribute:: models
      :type:  List[Model]

      The list of models that are supported.


.. py:class:: Slab

   Bases: :py:obj:`_DataModel`


   Stores all information about a slab that is returned from the API.


   .. py:attribute:: atoms
      :type:  Atoms

      The structure of the slab.


   .. py:attribute:: metadata
      :type:  SlabMetadata

      Extra information about the slab.


.. py:class:: SlabMetadata

   Bases: :py:obj:`_DataModel`


   Stores metadata about a slab that is returned from the API.


   .. py:attribute:: bulk_src_id
      :type:  str

      The ID of the bulk material from which the slab was derived.


   .. py:attribute:: millers
      :type:  Tuple[int, int, int]

      The Miller indices of the slab relative to bulk structure.


   .. py:attribute:: shift
      :type:  float

      The position along the vector defined by the Miller indices at which a
      cut was taken to generate the slab surface.


   .. py:attribute:: top
      :type:  bool

      If False, the top and bottom surfaces for this millers/shift pair are
      distinct and this slab represents the bottom surface.


.. py:class:: Slabs

   Bases: :py:obj:`_DataModel`


   Stores the response from a request to fetch slabs for a bulk structure.


   .. py:attribute:: slabs
      :type:  List[Slab]

      The list of slabs that were generated from the input bulk structure.


.. py:class:: Status(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Relaxation status of a single adsorbate placement on a slab.


   .. py:attribute:: NOT_AVAILABLE
      :value: 'not_available'


      The configuration exists but the result is not yet available. It is
      possible that checking again in the future could yield a result.


   .. py:attribute:: FAILED_RELAXATION
      :value: 'failed_relaxation'


      The relaxation failed for this configuration.


   .. py:attribute:: SUCCESS
      :value: 'success'


      The relaxation was successful and the requested information about the
      configuration was returned.


   .. py:attribute:: DOES_NOT_EXIST
      :value: 'does_not_exist'


      The requested configuration does not exist.


   .. py:method:: __str__() -> str


.. py:function:: get_results_ui_url(api_host: str, system_id: str) -> str | None

   Generates the URL at which results for the input system can be
   visualized.

   :param api_host: The API host on which the system was run.
   :param system_id: ID of the system being visualized.

   :returns: The URL at which the input system can be visualized. None if the
             API host is not recognized.


.. py:class:: AdsorbateBindingSites

   Stores the inputs and results of a set of relaxations of adsorbate
   placements on the surface of a slab.


   .. py:attribute:: adsorbate
      :type:  str

      Description of the adsorbate.


   .. py:attribute:: bulk
      :type:  fairchem.demo.ocpapi.client.Bulk

      The bulk material that was being modeled.


   .. py:attribute:: model
      :type:  str

      The type of the model that was run.


   .. py:attribute:: slabs
      :type:  list[AdsorbateSlabRelaxations]

      The list of slabs that were generated from the bulk structure. Each
      contains its own list of adsorbate placements.


.. py:class:: AdsorbateSlabRelaxations

   Stores the relaxations of adsorbate placements on the surface of a slab.


   .. py:attribute:: slab
      :type:  fairchem.demo.ocpapi.client.Slab

      The slab on which the adsorbate was placed.


   .. py:attribute:: configs
      :type:  list[fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationResult]

      Details of the relaxation of each adsorbate placement, including the
      final position.


   .. py:attribute:: system_id
      :type:  str

      The ID of the system that stores all of the relaxations.


   .. py:attribute:: api_host
      :type:  str

      The API host on which the relaxations were run.


   .. py:attribute:: ui_url
      :type:  str | None

      The URL at which results can be visualized.


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


.. py:exception:: UnsupportedAdsorbateException(adsorbate: str)

   Bases: :py:obj:`AdsorbatesException`


   Exception raised when an adsorbate is not supported in the API.


.. py:exception:: UnsupportedBulkException(bulk: str)

   Bases: :py:obj:`AdsorbatesException`


   Exception raised when a bulk material is not supported in the API.


.. py:exception:: UnsupportedModelException(model: str, allowed_models: list[str])

   Bases: :py:obj:`AdsorbatesException`


   Exception raised when a model is not supported in the API.


.. py:function:: find_adsorbate_binding_sites(adsorbate: str, bulk: str, model: str = 'equiformer_v2_31M_s2ef_all_md', adslab_filter: Callable[[list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]], Awaitable[list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]]] = _DEFAULT_ADSLAB_FILTER, client: fairchem.demo.ocpapi.client.Client = DEFAULT_CLIENT, lifetime: Lifetime = Lifetime.SAVE) -> AdsorbateBindingSites
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


.. py:function:: get_adsorbate_slab_relaxation_results(system_id: str, config_ids: list[int] | None = None, fields: list[str] | None = None, client: fairchem.demo.ocpapi.client.Client = DEFAULT_CLIENT) -> list[fairchem.demo.ocpapi.client.AdsorbateSlabRelaxationResult]
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


.. py:function:: wait_for_adsorbate_slab_relaxations(system_id: str, check_immediately: bool = False, slow_interval_sec: float = 30, fast_interval_sec: float = 10, pbar: tqdm.tqdm | None = None, client: fairchem.demo.ocpapi.client.Client = DEFAULT_CLIENT) -> dict[int, fairchem.demo.ocpapi.client.Status]
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


.. py:class:: keep_all_slabs

   Adslab filter than returns all slabs.


   .. py:method:: __call__(adslabs: list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]) -> list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]
      :async:



.. py:class:: keep_slabs_with_miller_indices(miller_indices: Iterable[tuple[int, int, int]])

   Adslab filter that keeps any slabs with the configured miller indices.
   Slabs with other miller indices will be ignored.


   .. py:attribute:: _unique_millers
      :type:  set[tuple[int, int, int]]


   .. py:method:: __call__(adslabs: list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]) -> list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]
      :async:



.. py:class:: prompt_for_slabs_to_keep

   Adslab filter than presents the user with an interactive prompt to choose
   which of the input slabs to keep.


   .. py:method:: _sort_key(adslab: fairchem.demo.ocpapi.client.AdsorbateSlabConfigs) -> tuple[tuple[int, int, int], float, str]
      :staticmethod:


      Generates a sort key from the input adslab. Returns the miller indices,
      shift, and top/bottom label so that they will be sorted by those values
      in that order.



   .. py:method:: __call__(adslabs: list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]) -> list[fairchem.demo.ocpapi.client.AdsorbateSlabConfigs]
      :async:



.. py:data:: NO_LIMIT
   :type:  NoLimitType
   :value: 0


.. py:data:: NoLimitType

.. py:class:: RateLimitLogging

   Controls logging when rate limits are hit.


   .. py:attribute:: logger
      :type:  logging.Logger

      The logger to use.


   .. py:attribute:: action
      :type:  str

      A short description of the action being attempted.


.. py:function:: retry_api_calls(max_attempts: int | NoLimitType = 3, rate_limit_logging: RateLimitLogging | None = None, fixed_wait_sec: float = 2, max_jitter_sec: float = 1) -> Any

   Decorator with sensible defaults for retrying calls to the OCP API.

   :param max_attempts: The maximum number of calls to make. If NO_LIMIT,
                        retries will be made forever.
   :param rate_limit_logging: If not None, log statements will be generated
                              using this configuration when a rate limit is hit.
   :param fixed_wait_sec: The fixed number of seconds to wait when retrying an
                          exception that does *not* include a retry-after value. The default
                          value is sensible; this is exposed mostly for testing.
   :param max_jitter_sec: The maximum number of seconds that will be randomly
                          added to wait times. The default value is sensible; this is exposed
                          mostly for testing.


