:py:mod:`ocpapi.client.models`
==============================

.. py:module:: ocpapi.client.models


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpapi.client.models._DataModel
   ocpapi.client.models.Model
   ocpapi.client.models.Models
   ocpapi.client.models.Bulk
   ocpapi.client.models.Bulks
   ocpapi.client.models.Adsorbates
   ocpapi.client.models.Atoms
   ocpapi.client.models.SlabMetadata
   ocpapi.client.models.Slab
   ocpapi.client.models.Slabs
   ocpapi.client.models.AdsorbateSlabConfigs
   ocpapi.client.models.AdsorbateSlabRelaxationsSystem
   ocpapi.client.models.AdsorbateSlabRelaxationsRequest
   ocpapi.client.models.Status
   ocpapi.client.models.AdsorbateSlabRelaxationResult
   ocpapi.client.models.AdsorbateSlabRelaxationsResults




.. py:class:: _DataModel


   Base class for all data models.

   .. py:attribute:: other_fields
      :type: dataclasses_json.CatchAll

      Fields that may have been added to the API that all not yet supported
      explicitly in this class.


.. py:class:: Model


   Bases: :py:obj:`_DataModel`

   Stores information about a single model supported in the API.

   .. py:attribute:: id
      :type: str

      The ID of the model.


.. py:class:: Models


   Bases: :py:obj:`_DataModel`

   Stores the response from a request for models supported in the API.

   .. py:attribute:: models
      :type: List[Model]

      The list of models that are supported.


.. py:class:: Bulk


   Bases: :py:obj:`_DataModel`

   Stores information about a single bulk material.

   .. py:attribute:: src_id
      :type: str

      The ID of the material.

   .. py:attribute:: formula
      :type: str

      The chemical formula of the material.

   .. py:attribute:: elements
      :type: List[str]

      The list of elements in the material.


.. py:class:: Bulks


   Bases: :py:obj:`_DataModel`

   Stores the response from a request to fetch bulks supported in the API.

   .. py:attribute:: bulks_supported
      :type: List[Bulk]

      List of bulks that can be used in the API.


.. py:class:: Adsorbates


   Bases: :py:obj:`_DataModel`

   Stores the response from a request to fetch adsorbates supported in the
   API.

   .. py:attribute:: adsorbates_supported
      :type: List[str]

      List of adsorbates that can be used in the API.


.. py:class:: Atoms


   Bases: :py:obj:`_DataModel`

   Subset of the fields from an ASE Atoms object that are used within this
   API.

   .. py:attribute:: cell
      :type: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]

      3x3 matrix with unit cell vectors.

   .. py:attribute:: pbc
      :type: Tuple[bool, bool, bool]

      Whether the structure is periodic along the a, b, and c lattice vectors,
      respectively.

   .. py:attribute:: numbers
      :type: List[int]

      The atomic number of each atom in the unit cell.

   .. py:attribute:: positions
      :type: List[Tuple[float, float, float]]

      The coordinates of each atom in the unit cell, relative to the cartesian
      frame.

   .. py:attribute:: tags
      :type: List[int]

      Labels for each atom in the unit cell where 0 represents a subsurface atom
      (fixed during optimization), 1 represents a surface atom, and 2 represents
      an adsorbate atom.

   .. py:method:: to_ase_atoms() -> ase.Atoms

      Creates an ase.Atoms object with the positions, element numbers,
      etc. populated from values on this object.

      :returns: ase.Atoms object with values from this object.



.. py:class:: SlabMetadata


   Bases: :py:obj:`_DataModel`

   Stores metadata about a slab that is returned from the API.

   .. py:attribute:: bulk_src_id
      :type: str

      The ID of the bulk material from which the slab was derived.

   .. py:attribute:: millers
      :type: Tuple[int, int, int]

      The Miller indices of the slab relative to bulk structure.

   .. py:attribute:: shift
      :type: float

      The position along the vector defined by the Miller indices at which a
      cut was taken to generate the slab surface.

   .. py:attribute:: top
      :type: bool

      If False, the top and bottom surfaces for this millers/shift pair are
      distinct and this slab represents the bottom surface.


.. py:class:: Slab


   Bases: :py:obj:`_DataModel`

   Stores all information about a slab that is returned from the API.

   .. py:attribute:: atoms
      :type: Atoms

      The structure of the slab.

   .. py:attribute:: metadata
      :type: SlabMetadata

      Extra information about the slab.


.. py:class:: Slabs


   Bases: :py:obj:`_DataModel`

   Stores the response from a request to fetch slabs for a bulk structure.

   .. py:attribute:: slabs
      :type: List[Slab]

      The list of slabs that were generated from the input bulk structure.


.. py:class:: AdsorbateSlabConfigs


   Bases: :py:obj:`_DataModel`

   Stores the response from a request to fetch placements of a single
   absorbate on a slab.

   .. py:attribute:: adsorbate_configs
      :type: List[Atoms]

      List of structures, each representing one possible adsorbate placement.

   .. py:attribute:: slab
      :type: Slab

      The structure of the slab on which the adsorbate is placed.


.. py:class:: AdsorbateSlabRelaxationsSystem


   Bases: :py:obj:`_DataModel`

   Stores the response from a request to submit a new batch of adsorbate
   slab relaxations.

   .. py:attribute:: system_id
      :type: str

      Unique ID for this set of relaxations which can be used to fetch results
      later.

   .. py:attribute:: config_ids
      :type: List[int]

      The list of IDs assigned to each of the input adsorbate placements, in the
      same order in which they were submitted.


.. py:class:: AdsorbateSlabRelaxationsRequest


   Bases: :py:obj:`_DataModel`

   Stores the request to submit a new batch of adsorbate slab relaxations.

   .. py:attribute:: adsorbate
      :type: str

      Description of the adsorbate.

   .. py:attribute:: adsorbate_configs
      :type: List[Atoms]

      List of adsorbate placements being relaxed.

   .. py:attribute:: bulk
      :type: Bulk

      Information about the original bulk structure used to create the slab.

   .. py:attribute:: slab
      :type: Slab

      The structure of the slab on which adsorbates are placed.

   .. py:attribute:: model
      :type: str

      The type of the ML model being used during relaxations.

   .. py:attribute:: ephemeral
      :type: Optional[bool]

      Whether the relaxations can be deleted (assume they cannot be deleted if
      None).

   .. py:attribute:: adsorbate_reaction
      :type: Optional[str]

      If possible, an html-formatted string describing the reaction will be added
      to this field.


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

      Return str(self).



.. py:class:: AdsorbateSlabRelaxationResult


   Bases: :py:obj:`_DataModel`

   Stores information about a single adsorbate slab configuration, including
   outputs for the model used in relaxations.

   The API to fetch relaxation results supports requesting a subset of fields
   in order to limit the size of response payloads. Optional attributes will
   be defined only if they are including the response.

   .. py:attribute:: config_id
      :type: int

      ID of the configuration within the system.

   .. py:attribute:: status
      :type: Status

      The status of the request for information about this configuration.

   .. py:attribute:: system_id
      :type: Optional[str]

      The ID of the system in which the configuration was originally submitted.

   .. py:attribute:: cell
      :type: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]

      3x3 matrix with unit cell vectors.

   .. py:attribute:: pbc
      :type: Optional[Tuple[bool, bool, bool]]

      Whether the structure is periodic along the a, b, and c lattice vectors,
      respectively.

   .. py:attribute:: numbers
      :type: Optional[List[int]]

      The atomic number of each atom in the unit cell.

   .. py:attribute:: positions
      :type: Optional[List[Tuple[float, float, float]]]

      The coordinates of each atom in the unit cell, relative to the cartesian
      frame.

   .. py:attribute:: tags
      :type: Optional[List[int]]

      Labels for each atom in the unit cell where 0 represents a subsurface atom
      (fixed during optimization), 1 represents a surface atom, and 2 represents
      an adsorbate atom.

   .. py:attribute:: energy
      :type: Optional[float]

      The energy of the configuration.

   .. py:attribute:: energy_trajectory
      :type: Optional[List[float]]

      The energy of the configuration at each point along the relaxation
      trajectory.

   .. py:attribute:: forces
      :type: Optional[List[Tuple[float, float, float]]]

      The forces on each atom in the relaxed structure.

   .. py:method:: to_ase_atoms() -> ase.Atoms

      Creates an ase.Atoms object with the positions, element numbers,
      etc. populated from values on this object.

      The predicted energy and forces will also be copied to the new
      ase.Atoms object as a SinglePointCalculator (a calculator that
      stores the results of an already-run simulation).

      :returns: ase.Atoms object with values from this object.



.. py:class:: AdsorbateSlabRelaxationsResults


   Bases: :py:obj:`_DataModel`

   Stores the response from a request for results of adsorbate slab
   relaxations.

   .. py:attribute:: configs
      :type: List[AdsorbateSlabRelaxationResult]

      List of configurations in the system, each representing one placement of
      an adsorbate on a slab surface.

   .. py:attribute:: omitted_config_ids
      :type: List[int]

      List of IDs of configurations that were requested but omitted by the
      server. Results for these IDs can be requested again.


