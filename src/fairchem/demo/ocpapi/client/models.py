from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from dataclasses_json import CatchAll, Undefined, config, dataclass_json


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class _DataModel:
    """
    Base class for all data models.
    """

    other_fields: CatchAll
    """
    Fields that may have been added to the API that all not yet supported 
    explicitly in this class.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Model(_DataModel):
    """
    Stores information about a single model supported in the API.
    """

    id: str
    """
    The ID of the model.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Models(_DataModel):
    """
    Stores the response from a request for models supported in the API.
    """

    models: List[Model]
    """
    The list of models that are supported.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Bulk(_DataModel):
    """
    Stores information about a single bulk material.
    """

    src_id: str
    """
    The ID of the material.
    """

    formula: str
    """
    The chemical formula of the material.
    """

    # Stored under "els" in the API response
    elements: List[str] = field(metadata=config(field_name="els"))
    """
    The list of elements in the material.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Bulks(_DataModel):
    """
    Stores the response from a request to fetch bulks supported in the API.
    """

    bulks_supported: List[Bulk]
    """
    List of bulks that can be used in the API.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Adsorbates(_DataModel):
    """
    Stores the response from a request to fetch adsorbates supported in the
    API.
    """

    adsorbates_supported: List[str]
    """
    List of adsorbates that can be used in the API.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Atoms(_DataModel):
    """
    Subset of the fields from an ASE Atoms object that are used within this
    API.
    """

    cell: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]
    """
    3x3 matrix with unit cell vectors.
    """

    pbc: Tuple[bool, bool, bool]
    """
    Whether the structure is periodic along the a, b, and c lattice vectors, 
    respectively.
    """

    numbers: List[int]
    """
    The atomic number of each atom in the unit cell.
    """

    positions: List[Tuple[float, float, float]]
    """
    The coordinates of each atom in the unit cell, relative to the cartesian 
    frame.
    """

    tags: List[int]
    """
    Labels for each atom in the unit cell where 0 represents a subsurface atom 
    (fixed during optimization), 1 represents a surface atom, and 2 represents 
    an adsorbate atom.
    """

    def to_ase_atoms(self) -> "ASEAtoms":
        """
        Creates an ase.Atoms object with the positions, element numbers,
        etc. populated from values on this object.

        Returns:
            ase.Atoms object with values from this object.
        """

        from ase import Atoms as ASEAtoms
        from ase.constraints import FixAtoms

        return ASEAtoms(
            cell=self.cell,
            pbc=self.pbc,
            numbers=self.numbers,
            positions=self.positions,
            tags=self.tags,
            # Fix sub-surface atoms
            constraint=FixAtoms(mask=[t == 0 for t in self.tags]),
        )


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class SlabMetadata(_DataModel):
    """
    Stores metadata about a slab that is returned from the API.
    """

    # Stored under "bulk_id" in the API response
    bulk_src_id: str = field(metadata=config(field_name="bulk_id"))
    """
    The ID of the bulk material from which the slab was derived.
    """

    millers: Tuple[int, int, int]
    """
    The Miller indices of the slab relative to bulk structure.
    """

    shift: float
    """
    The position along the vector defined by the Miller indices at which a 
    cut was taken to generate the slab surface.
    """

    top: bool
    """
    If False, the top and bottom surfaces for this millers/shift pair are 
    distinct and this slab represents the bottom surface.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Slab(_DataModel):
    """
    Stores all information about a slab that is returned from the API.
    """

    # Stored under "slab_atomsobject" in the API response
    atoms: Atoms = field(metadata=config(field_name="slab_atomsobject"))
    """
    The structure of the slab.
    """

    # Stored under "slab_metadata" in the API response
    metadata: SlabMetadata = field(metadata=config(field_name="slab_metadata"))
    """
    Extra information about the slab.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Slabs(_DataModel):
    """
    Stores the response from a request to fetch slabs for a bulk structure.
    """

    slabs: List[Slab]
    """
    The list of slabs that were generated from the input bulk structure.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbateSlabConfigs(_DataModel):
    """
    Stores the response from a request to fetch placements of a single
    absorbate on a slab.
    """

    adsorbate_configs: List[Atoms]
    """
    List of structures, each representing one possible adsorbate placement.
    """

    slab: Slab
    """
    The structure of the slab on which the adsorbate is placed.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbateSlabRelaxationsSystem(_DataModel):
    """
    Stores the response from a request to submit a new batch of adsorbate
    slab relaxations.
    """

    system_id: str
    """
    Unique ID for this set of relaxations which can be used to fetch results 
    later.
    """

    config_ids: List[int]
    """
    The list of IDs assigned to each of the input adsorbate placements, in the
    same order in which they were submitted.
    """


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbateSlabRelaxationsRequest(_DataModel):
    """
    Stores the request to submit a new batch of adsorbate slab relaxations.
    """

    adsorbate: str
    """
    Description of the adsorbate.
    """

    adsorbate_configs: List[Atoms]
    """
    List of adsorbate placements being relaxed.
    """

    bulk: Bulk
    """
    Information about the original bulk structure used to create the slab.
    """

    slab: Slab
    """
    The structure of the slab on which adsorbates are placed.
    """

    model: str
    """
    The type of the ML model being used during relaxations.
    """

    # Omit from serialization when None
    ephemeral: Optional[bool] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    Whether the relaxations can be deleted (assume they cannot be deleted if 
    None).
    """

    adsorbate_reaction: Optional[str] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    If possible, an html-formatted string describing the reaction will be added 
    to this field.
    """


class Status(Enum):
    """
    Relaxation status of a single adsorbate placement on a slab.
    """

    NOT_AVAILABLE = "not_available"
    """
    The configuration exists but the result is not yet available. It is 
    possible that checking again in the future could yield a result.
    """

    FAILED_RELAXATION = "failed_relaxation"
    """
    The relaxation failed for this configuration.
    """

    SUCCESS = "success"
    """
    The relaxation was successful and the requested information about the 
    configuration was returned.
    """

    DOES_NOT_EXIST = "does_not_exist"
    """
    The requested configuration does not exist.
    """

    def __str__(self) -> str:
        return self.value


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbateSlabRelaxationResult(_DataModel):
    """
    Stores information about a single adsorbate slab configuration, including
    outputs for the model used in relaxations.

    The API to fetch relaxation results supports requesting a subset of fields
    in order to limit the size of response payloads. Optional attributes will
    be defined only if they are including the response.
    """

    config_id: int
    """
    ID of the configuration within the system.
    """

    status: Status
    """
    The status of the request for information about this configuration.
    """

    # Omit from serialization when None
    system_id: Optional[str] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    The ID of the system in which the configuration was originally submitted.
    """

    cell: Optional[
        Tuple[
            Tuple[float, float, float],
            Tuple[float, float, float],
            Tuple[float, float, float],
        ]
    ] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    3x3 matrix with unit cell vectors.
    """

    pbc: Optional[Tuple[bool, bool, bool]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    Whether the structure is periodic along the a, b, and c lattice vectors,
    respectively.
    """

    numbers: Optional[List[int]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    The atomic number of each atom in the unit cell.
    """

    positions: Optional[List[Tuple[float, float, float]]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    The coordinates of each atom in the unit cell, relative to the cartesian
    frame.
    """

    tags: Optional[List[int]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    Labels for each atom in the unit cell where 0 represents a subsurface atom
    (fixed during optimization), 1 represents a surface atom, and 2 represents
    an adsorbate atom.
    """

    energy: Optional[float] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    The energy of the configuration.
    """

    energy_trajectory: Optional[List[float]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    The energy of the configuration at each point along the relaxation
    trajectory.
    """

    forces: Optional[List[Tuple[float, float, float]]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    """
    The forces on each atom in the relaxed structure.
    """

    def to_ase_atoms(self) -> "ASEAtoms":
        """
        Creates an ase.Atoms object with the positions, element numbers,
        etc. populated from values on this object.

        The predicted energy and forces will also be copied to the new
        ase.Atoms object as a SinglePointCalculator (a calculator that
        stores the results of an already-run simulation).

        Returns:
            ase.Atoms object with values from this object.
        """
        from ase import Atoms as ASEAtoms
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.constraints import FixAtoms

        atoms: ASEAtoms = ASEAtoms(
            cell=self.cell,
            pbc=self.pbc,
            numbers=self.numbers,
            positions=self.positions,
            tags=self.tags,
        )
        if self.tags is not None:
            # Fix sub-surface atoms
            atoms.constraints = FixAtoms(mask=[t == 0 for t in self.tags])
        atoms.calc = SinglePointCalculator(
            atoms=atoms,
            energy=self.energy,
            forces=self.forces,
        )
        return atoms


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbateSlabRelaxationsResults(_DataModel):
    """
    Stores the response from a request for results of adsorbate slab
    relaxations.
    """

    configs: List[AdsorbateSlabRelaxationResult]
    """
    List of configurations in the system, each representing one placement of 
    an adsorbate on a slab surface.
    """

    omitted_config_ids: List[int] = field(default_factory=lambda: list())
    """
    List of IDs of configurations that were requested but omitted by the 
    server. Results for these IDs can be requested again.
    """
