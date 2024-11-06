ocpapi.client
=============

.. py:module:: ocpapi.client


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/ocpapi/client/client/index
   /autoapi/ocpapi/client/models/index
   /autoapi/ocpapi/client/ui/index


Exceptions
----------

.. autoapisummary::

   ocpapi.client.NonRetryableRequestException
   ocpapi.client.RateLimitExceededException
   ocpapi.client.RequestException


Classes
-------

.. autoapisummary::

   ocpapi.client.Client
   ocpapi.client.Adsorbates
   ocpapi.client.AdsorbateSlabConfigs
   ocpapi.client.AdsorbateSlabRelaxationResult
   ocpapi.client.AdsorbateSlabRelaxationsRequest
   ocpapi.client.AdsorbateSlabRelaxationsResults
   ocpapi.client.AdsorbateSlabRelaxationsSystem
   ocpapi.client.Atoms
   ocpapi.client.Bulk
   ocpapi.client.Bulks
   ocpapi.client.Model
   ocpapi.client.Models
   ocpapi.client.Slab
   ocpapi.client.SlabMetadata
   ocpapi.client.Slabs
   ocpapi.client.Status


Functions
---------

.. autoapisummary::

   ocpapi.client.get_results_ui_url


Package Contents
----------------

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


