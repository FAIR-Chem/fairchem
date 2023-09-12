from dataclasses import dataclass, field
from typing import List, Tuple

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
class BulksResponse(_Model):
    """
    Stores the response from a request to fetch bulks supported in the API.

    Attributes:
        bulks_supported: List of bulks that can be used in the API.
    """

    bulks_supported: List[Bulk]


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbatesResponse(_Model):
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
class SlabsResponse(_Model):
    """
    Stores the response from a request to fetch slabs for a bulk structure.

    Attributes:
        slabs: The list of slabs that were generated from the input bulk
            structure.
    """

    slabs: List[Slab]


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class AdsorbateSlabConfigsResponse(_Model):
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
class AdsorbateSlabRelaxationsResponse(_Model):
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
