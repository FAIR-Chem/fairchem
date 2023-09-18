from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from dataclasses_json import CatchAll, Undefined, config, dataclass_json


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class _Model:
    """
    Base class for all data models.

    Attributes:
        other: Fields that may have been added to the API that all not yet
            supported explicitly in this class.
    """

    other_fields: CatchAll


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Bulk(_Model):
    """
    Stores information about a single bulk material.

    Attributes:
        src_id: The ID of the material.
        formula: The chemical formula of the material.
        elements: The list of elements in the material.
    """

    src_id: str
    formula: str
    # Stored under "els" in the API response
    elements: List[str] = field(metadata=config(field_name="els"))


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Bulks(_Model):
    """
    Stores the response from a request to fetch bulks supported in the API.

    Attributes:
        bulks_supported: List of bulks that can be used in the API.
    """

    bulks_supported: List[Bulk]


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Adsorbates(_Model):
    """
    Stores the response from a request to fetch adsorbates supported in the
    API.

    Attributes:
        adsorbates_supported: List of adsorbates that can be used in the API,
            each in SMILES notation.
    """

    adsorbates_supported: List[str]


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Atoms(_Model):
    """
    Subset of the fields from an ASE Atoms object that are used within this
    API.

    Attributes:
        cell: 3x3 matrix with unit cell vectors.
        pbc: Whether the structure is periodic along the a, b, and c lattice
            vectors, respectively.
        numbers: The atomic number of each atom in the unit cell.
        positions: The coordinates of each atom in the unit cell, relative to
            the cartesian frame.
        tags: Labels for each atom in the unit cell where 0 represents a
            subsurface atom (fixed during optimization), 1 represents a
            surface atom, and 2 represents an adsorbate atom.
    """

    cell: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]
    pbc: Tuple[bool, bool, bool]
    numbers: List[int]
    positions: List[Tuple[float, float, float]]
    tags: List[int]


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class SlabMetadata(_Model):
    """
    Stores metadata about a slab that is returned from the API.

    Attributes:
        bulk_src_id: The ID of the bulk material from which the slab was
            derived.
        millers: The Miller indices of the slab relative to bulk structure.
        shift: The position along the vector defined by the Miller indices
            at which a cut was taken to generate the slab surface.
        top: If False, the top and bottom surfaces for this millers/shift
            pair are distinct and this slab represents the bottom surface.
    """

    # Stored under "bulk_id" in the API response
    bulk_src_id: str = field(metadata=config(field_name="bulk_id"))
    millers: Tuple[int, int, int]
    shift: float
    top: bool


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Slab(_Model):
    """
    Stores all information about a slab that is returned from the API.

    Attributes:
        atoms: The structure of the slab.
        metadata: Extra information about the slab.
    """

    # Stored under "slab_atomsobject" in the API response
    atoms: Atoms = field(metadata=config(field_name="slab_atomsobject"))
    # Stored under "slab_metadata" in the API response
    metadata: SlabMetadata = field(metadata=config(field_name="slab_metadata"))


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class Slabs(_Model):
    """
    Stores the response from a request to fetch slabs for a bulk structure.

    Attributes:
        slabs: The list of slabs that were generated from the input bulk
            structure.
    """

    slabs: List[Slab]


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbateSlabConfigs(_Model):
    """
    Stores the response from a request to fetch placements of a single
    absorbate on a slab.

    Attributes:
        adsorbate_configs: List of structures, each representing one possible
            adsorbate placement.
        slab: The structure of the slab on which the adsorbate is placed.
    """

    adsorbate_configs: List[Atoms]
    slab: Slab


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbateSlabRelaxationsSystem(_Model):
    """
    Stores the response from a request to submit a new batch of adsorbate
    slab relaxations.

    Attributes:
        system_id: Unique ID for this set of relaxations which can be used to
            fetch results later.
        config_ids: The list of IDs assigned to each of the input adsorbate
            placements, in the same order in which they were submitted.
    """

    system_id: str
    config_ids: List[int]


class Status(Enum):
    """
    Relaxation status of a single adsorbate placement on a slab.

    Attributes:
        NOT_AVAILABLE: The configuration exists but the result is not yet
            available. It is possible that checking again in the future could
            yield a result.
        FAILED_RELAXATION: The relaxation failed for this configuration.
        SUCCESS: The relaxation was successful and the requested information
            about the configuration was returned.
        DOES_NOT_EXIST: The requested configuration does not exist.
        OMITTED: The requestion configuration exists but information about it
            was not included in the response. This can happen when the response
            becomes too large and a subset of the configurations were left out.
    """

    NOT_AVAILABLE = "not_available"
    FAILED_RELAXATION = "failed_relaxation"
    SUCCESS = "success"
    DOES_NOT_EXIST = "does_not_exist"
    OMITTED = "omitted"

    def __str__(self) -> str:
        return self.value


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbateSlabRelaxationResult(_Model):
    """
    Stores information about a single adsorbate slab configuration, including
    outputs for the model used in relaxations.

    The API to fetch relaxation results supports requesting a subset of fields
    in order to limit the size of response payloads. Optional attributes will
    be defined only if they are including the response.

    Attributes:
        config_id: ID of the configuration within the system
        status: The status of the request for information about this
            configuration.
        system_id: The ID of the system in which the configuration was
            originally submitted.
        cell: 3x3 matrix with unit cell vectors.
        pbc: Whether the structure is periodic along the a, b, and c lattice
            vectors, respectively.
        numbers: The atomic number of each atom in the unit cell.
        positions: The coordinates of each atom in the unit cell, relative to
            the cartesian frame.
        tags: Labels for each atom in the unit cell where 0 represents a
            subsurface atom (fixed during optimization), 1 represents a
            surface atom, and 2 represents an adsorbate atom.
        energy: The energy of the configuration.
        energy_trajectory: The energy of the configuration at each point along
            the relaxation trajectory.
        forces: The forces on each atom in the relaxed structure.
    """

    config_id: int
    status: Status
    # Omit from serialization when None
    system_id: Optional[str] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
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
    pbc: Optional[Tuple[bool, bool, bool]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    numbers: Optional[List[int]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    positions: Optional[List[Tuple[float, float, float]]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    tags: Optional[List[int]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    energy: Optional[float] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    energy_trajectory: Optional[float] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )
    forces: Optional[List[Tuple[float, float, float]]] = field(
        default=None,
        metadata=config(exclude=lambda v: v is None),
    )


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbateSlabRelaxationsResults(_Model):
    """
    Stores the response from a request for results of adsorbate slab
    relaxations.

    Attributes:
        configs: List of configurations in the system, each representing
            one placement of an adsorbate on a slab surface.
    """

    configs: List[AdsorbateSlabRelaxationResult]
