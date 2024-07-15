from __future__ import annotations

import os
import pickle
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from fairchem.data.oc.core.slab import Slab
from fairchem.data.oc.databases.pkls import BULK_PKL_PATH

if TYPE_CHECKING:
    import ase


class Bulk:
    """
    Initializes a bulk object in one of 4 ways:
    - Directly pass in an ase.Atoms object.
    - Pass in index of bulk to select from bulk database.
    - Pass in the src_id of the bulk to select from the bulk database.
    - Randomly sample a bulk from bulk database if no other option is passed.

    Arguments
    ---------
    bulk_atoms: ase.Atoms
        Bulk structure.
    bulk_id_from_db: int
        Index of bulk in database pkl to select.
    bulk_src_id_from_db: int
        Src id of bulk to select (e.g. "mp-30").
    bulk_db_path: str
        Path to bulk database.
    bulk_db: List[Dict[str, Any]]
        Already-loaded database.
    """

    def __init__(
        self,
        bulk_atoms: ase.Atoms = None,
        bulk_id_from_db: int | None = None,
        bulk_src_id_from_db: str | None = None,
        bulk_db_path: str = BULK_PKL_PATH,
        bulk_db: list[dict[str, Any]] | None = None,
    ):
        self.bulk_id_from_db = bulk_id_from_db
        self.bulk_db_path = bulk_db_path

        if bulk_atoms is not None:
            self.atoms = bulk_atoms.copy()
            self.src_id = None
        else:
            if bulk_db is None:
                with open(bulk_db_path, "rb") as fp:
                    bulk_db = pickle.load(fp)

            if bulk_id_from_db is not None:
                bulk_obj = bulk_db[bulk_id_from_db]
                self.atoms, self.src_id = bulk_obj["atoms"], bulk_obj["src_id"]
            elif bulk_src_id_from_db is not None:
                bulk_obj_tuple = [
                    (idx, bulk)
                    for idx, bulk in enumerate(bulk_db)
                    if bulk["src_id"] == bulk_src_id_from_db
                ]
                if len(bulk_obj_tuple) < 1:
                    warnings.warn(
                        "A bulk with that src id was not found. Choosing one at random instead"
                    )
                    self._get_bulk_from_random(bulk_db)
                else:
                    bulk_obj = bulk_obj_tuple[0][1]
                    self.bulk_id_from_db = bulk_obj_tuple[0][0]
                    self.atoms, self.src_id = bulk_obj["atoms"], bulk_obj["src_id"]
            else:
                self._get_bulk_from_random(bulk_db)

    def _get_bulk_from_random(self, bulk_db):
        self.bulk_id_from_db = np.random.randint(len(bulk_db))
        bulk_obj = bulk_db[self.bulk_id_from_db]
        self.atoms, self.src_id = bulk_obj["atoms"], bulk_obj["src_id"]

    def set_source_dataset_id(self, src_id: str):
        self.src_id = src_id

    def set_bulk_id_from_db(self, bulk_id_from_db: int):
        self.bulk_id_from_db = bulk_id_from_db

    def get_slabs(self, max_miller=2, precomputed_slabs_dir=None):
        """
        Returns a list of possible slabs for this bulk instance.
        """
        precomp_slabs_pkl = None
        if precomputed_slabs_dir is not None and self.bulk_id_from_db is not None:
            precomp_slabs_pkl = os.path.join(
                precomputed_slabs_dir, f"{self.bulk_id_from_db}.pkl"
            )

        if precomp_slabs_pkl is not None and os.path.exists(precomp_slabs_pkl):
            slabs = Slab.from_precomputed_slabs_pkl(
                self,
                precomp_slabs_pkl,
                max_miller=max_miller,
            )
        else:
            # If precomputed_slabs_dir was provided but the specific pkl for the
            # bulk doesn't exist, save it out after we generate the slabs.
            slabs = Slab.from_bulk_get_all_slabs(
                self, max_miller=max_miller, save_path=precomp_slabs_pkl
            )

        return slabs

    def __len__(self):
        return len(self.atoms)

    def __str__(self):
        return f"Bulk: ({self.atoms.get_chemical_formula()})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other) -> bool:
        return self.atoms == other.atoms
