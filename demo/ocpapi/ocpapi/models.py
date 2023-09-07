from dataclasses import dataclass, field
from typing import List

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

    src_id: str = ""
    formula: str = ""
    elements: List[str] = field(
        metadata=config(field_name="els"),
        default_factory=list,
    )


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class BulksResponse(_Model):
    """
    Stores the response from a request to fetch bulks supported in the API.

    Attributes:
        bulks_supported: List of bulks that can be used in the API.
    """

    bulks_supported: List[Bulk] = field(default_factory=list)


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

    adsorbates_supported: List[str] = field(default_factory=list)
