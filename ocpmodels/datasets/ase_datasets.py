import bisect
import copy
import functools
import glob
import logging
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import ase
import numpy as np
from torch import tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from ocpmodels.common.registry import registry
from ocpmodels.datasets.lmdb_database import LMDBDatabase
from ocpmodels.datasets.target_metadata_guesser import guess_property_metadata
from ocpmodels.preprocessing import AtomsToGraphs


def apply_one_tags(
    atoms: ase.Atoms, skip_if_nonzero: bool = True, skip_always: bool = False
):
    """
    This function will apply tags of 1 to an ASE atoms object.
    It is used as an atoms_transform in the datasets contained in this file.

    Certain models will treat atoms differently depending on their tags.
    For example, GemNet-OC by default will only compute triplet and quadruplet interactions
    for atoms with non-zero tags. This model throws an error if there are no tagged atoms.
    For this reason, the default behavior is to tag atoms in structures with no tags.

    args:
        skip_if_nonzero (bool): If at least one atom has a nonzero tag, do not tag any atoms

        skip_always (bool): Do not apply any tags. This arg exists so that this function can be disabled
                without needing to pass a callable (which is currently difficult to do with main.py)
    """
    if skip_always:
        return atoms

    if np.all(atoms.get_tags() == 0) or not skip_if_nonzero:
        atoms.set_tags(np.ones(len(atoms)))

    return atoms


class AseAtomsDataset(Dataset, ABC):
    """
    This is an abstract Dataset that includes helpful utilities for turning
    ASE atoms objects into OCP-usable data objects. This should not be instantiated directly
    as get_atoms_object and load_dataset_get_ids are not implemented in this base class.

    Derived classes must add at least two things:
        self.get_atoms_object(id): a function that takes an identifier and returns a corresponding atoms object

        self.load_dataset_get_ids(config: dict): This function is responsible for any initialization/loads
            of the dataset and importantly must return a list of all possible identifiers that can be passed into
            self.get_atoms_object(id)

    Identifiers need not be any particular type.
    """

    def __init__(
        self, config, transform=None, atoms_transform=apply_one_tags
    ) -> None:
        self.config = config

        a2g_args = config.get("a2g_args", {})
        if a2g_args is None:
            a2g_args = {}

        # Make sure we always include PBC info in the resulting atoms objects
        a2g_args["r_pbc"] = True
        self.a2g = AtomsToGraphs(**a2g_args)

        self.transform = transform
        self.atoms_transform = atoms_transform

        if self.config.get("keep_in_memory", False):
            self.__getitem__ = functools.cache(self.__getitem__)

        self.ids = self.load_dataset_get_ids(config)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx):
        # Handle slicing
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self.ids)))]

        # Get atoms object via derived class method
        atoms = self.get_atoms_object(self.ids[idx])

        # Transform atoms object
        if self.atoms_transform is not None:
            atoms = self.atoms_transform(
                atoms, **self.config.get("atoms_transform_args", {})
            )

        sid = atoms.info.get("sid", self.ids[idx])
        try:
            sid = tensor([sid])
            warnings.warn(
                "Supplied sid is not numeric (or missing). Using dataset indices instead."
            )
        except:
            sid = tensor([idx])

        fid = atoms.info.get("fid", tensor([0]))

        # Convert to data object
        data_object = self.a2g.convert(atoms, sid)
        data_object.fid = fid

        # Transform data object
        if self.transform is not None:
            data_object = self.transform(
                data_object, **self.config.get("transform_args", {})
            )

        if self.config.get("include_relaxed_energy", False):
            data_object.y_relaxed = self.get_relaxed_energy(self.ids[idx])

        return data_object

    @abstractmethod
    def get_atoms_object(self, identifier):
        # This function should return an ASE atoms object.
        raise NotImplementedError(
            "Returns an ASE atoms object. Derived classes should implement this function."
        )

    @abstractmethod
    def load_dataset_get_ids(self, config):
        # This function should return a list of ids that can be used to index into the database
        raise NotImplementedError(
            "Every ASE dataset needs to declare a function to load the dataset and return a list of ids."
        )

    def close_db(self) -> None:
        # This method is sometimes called by a trainer
        pass

    def guess_target_metadata(self, num_samples: int = 100):
        metadata = {}

        if num_samples < len(self):
            metadata["targets"] = guess_property_metadata(
                [
                    self.get_atoms_object(self.ids[idx])
                    for idx in np.random.choice(
                        len(self), size=(num_samples,), replace=False
                    )
                ]
            )
        else:
            metadata["targets"] = guess_property_metadata(
                [
                    self.get_atoms_object(self.ids[idx])
                    for idx in range(len(self))
                ]
            )

        return metadata

    def get_metadata(self):
        return self.guess_target_metadata()


@registry.register_dataset("ase_read")
class AseReadDataset(AseAtomsDataset):
    """
    This Dataset uses ase.io.read to load data from a directory on disk.
    This is intended for small-scale testing and demonstrations of OCP.
    Larger datasets are better served by the efficiency of other dataset types
    such as LMDB.

    For a full list of ASE-readable filetypes, see
    https://wiki.fysik.dtu.dk/ase/ase/io/io.html

    args:
        config (dict):
            src (str): The source folder that contains your ASE-readable files

            pattern (str): Filepath matching each file you want to read
                    ex. "*/POSCAR", "*.cif", "*.xyz"
                    search recursively with two wildcards: "**/POSCAR" or "**/*.cif"

            a2g_args (dict): Keyword arguments for ocpmodels.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True and/or "r_forces":True as appropriate
                    In that case, energy/forces must be in the files you read (ex. OUTCAR)

            ase_read_args (dict): Keyword arguments for ase.io.read()

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            include_relaxed_energy (bool): Include the relaxed energy in the resulting data object.
                    The relaxed structure is assumed to be the final structure in the file
                    (e.g. the last frame of a .traj).

            atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

            transform_args (dict): Additional keyword arguments for the transform callable

        atoms_transform (callable, optional): Additional preprocessing function applied to the Atoms
                    object. Useful for applying tags, for example.

        transform (callable, optional): Additional preprocessing function for the Data object

    """

    def load_dataset_get_ids(self, config) -> List[Path]:
        self.ase_read_args = config.get("ase_read_args", {})

        if ":" in self.ase_read_args.get("index", ""):
            raise NotImplementedError(
                "To read multiple structures from a single file, please use AseReadMultiStructureDataset."
            )

        self.path = Path(config["src"])
        if self.path.is_file():
            raise Exception("The specified src is not a directory")

        if self.config.get("include_relaxed_energy", False):
            self.relaxed_ase_read_args = copy.deepcopy(self.ase_read_args)
            self.relaxed_ase_read_args["index"] = "-1"

        return list(self.path.glob(f'{config["pattern"]}'))

    def get_atoms_object(self, identifier):
        try:
            atoms = ase.io.read(identifier, **self.ase_read_args)
        except Exception as err:
            warnings.warn(f"{err} occured for: {identifier}")
            raise err

        return atoms

    def get_relaxed_energy(self, identifier):
        relaxed_atoms = ase.io.read(identifier, **self.relaxed_ase_read_args)
        return relaxed_atoms.get_potential_energy(apply_constraint=False)


@registry.register_dataset("ase_read_multi")
class AseReadMultiStructureDataset(AseAtomsDataset):
    """
    This Dataset can read multiple structures from each file using ase.io.read.
    The disadvantage is that all files must be read at startup.
    This is a significant cost for large datasets.

    This is intended for small-scale testing and demonstrations of OCP.
    Larger datasets are better served by the efficiency of other dataset types
    such as LMDB.

    For a full list of ASE-readable filetypes, see
    https://wiki.fysik.dtu.dk/ase/ase/io/io.html

    args:
        config (dict):
            src (str): The source folder that contains your ASE-readable files

            pattern (str): Filepath matching each file you want to read
                    ex. "*.traj", "*.xyz"
                    search recursively with two wildcards: "**/POSCAR" or "**/*.cif"

            index_file (str): Filepath to an indexing file, which contains each filename
                    and the number of structures contained in each file. For instance:

                    /path/to/relaxation1.traj 200
                    /path/to/relaxation2.traj 150

                    This will overrule the src and pattern that you specify!

            a2g_args (dict): Keyword arguments for ocpmodels.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True and/or "r_forces":True as appropriate
                    In that case, energy/forces must be in the files you read (ex. OUTCAR)

            ase_read_args (dict): Keyword arguments for ase.io.read()

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            include_relaxed_energy (bool): Include the relaxed energy in the resulting data object.
                    The relaxed structure is assumed to be the final structure in the file
                    (e.g. the last frame of a .traj).

            use_tqdm (bool): Use TQDM progress bar when initializing dataset

            atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

            transform_args (dict): Additional keyword arguments for the transform callable

        atoms_transform (callable, optional): Additional preprocessing function applied to the Atoms
            object. Useful for applying tags, for example.

        transform (callable, optional): Additional preprocessing function for the Data object
    """

    def load_dataset_get_ids(self, config):
        self.ase_read_args = config.get("ase_read_args", {})
        if not hasattr(self.ase_read_args, "index"):
            self.ase_read_args["index"] = ":"

        if config.get("index_file", None) is not None:
            f = open(config["index_file"], "r")
            index = f.readlines()

            ids = []
            for line in index:
                filename = line.split(" ")[0]
                for i in range(int(line.split(" ")[1])):
                    ids.append(f"{filename} {i}")

            return ids

        self.path = Path(config["src"])
        if self.path.is_file():
            raise Exception("The specified src is not a directory")
        filenames = list(self.path.glob(f'{config["pattern"]}'))

        ids = []

        if config.get("use_tqdm", True):
            filenames = tqdm(filenames)
        for filename in filenames:
            try:
                structures = ase.io.read(filename, **self.ase_read_args)
            except Exception as err:
                warnings.warn(f"{err} occured for: {filename}")
            else:
                for i, structure in enumerate(structures):
                    ids.append(f"{filename} {i}")

        return ids

    def get_atoms_object(self, identifier):
        try:
            atoms = ase.io.read(
                "".join(identifier.split(" ")[:-1]), **self.ase_read_args
            )[int(identifier.split(" ")[-1])]
        except Exception as err:
            warnings.warn(f"{err} occured for: {identifier}")
            raise err

        if "sid" not in atoms.info:
            atoms.info["sid"] = "".join(identifier.split(" ")[:-1])
        if "fid" not in atoms.info:
            atoms.info["fid"] = int(identifier.split(" ")[-1])

        return atoms

    def get_metadata(self):
        return {}

    def get_relaxed_energy(self, identifier):
        relaxed_atoms = ase.io.read(
            "".join(identifier.split(" ")[:-1]), **self.ase_read_args
        )[-1]
        return relaxed_atoms.get_potential_energy(apply_constraint=False)


class dummy_list(list):
    def __init__(self, max) -> None:
        self.max = max
        return

    def __len__(self):
        return self.max

    def __getitem__(self, idx):
        # Handle slicing
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self.max))]

        # Cast idx as int since it could be a tensor index
        idx = int(idx)

        # Handle negative indices (referenced from end)
        if idx < 0:
            idx += self.max

        if 0 <= idx < self.max:
            return idx
        else:
            raise IndexError


@registry.register_dataset("ase_db")
class AseDBDataset(AseAtomsDataset):
    """
    This Dataset connects to an ASE Database, allowing the storage of atoms objects
    with a variety of backends including JSON, SQLite, and database server options.

    For more information, see:
    https://databases.fysik.dtu.dk/ase/ase/db/db.html

    args:
        config (dict):
            src (str): Either
                    - the path an ASE DB,
                    - the connection address of an ASE DB,
                    - a folder with multiple ASE DBs,
                    - a glob string to use to find ASE DBs, or
                    - a list of ASE db paths/addresses.
                    If a folder, every file will be attempted as an ASE DB, and warnings
                    are raised for any files that can't connect cleanly

                    Note that for large datasets, ID loading can be slow and there can be many
                    ids, so it's advised to make loading the id list as easy as possible. There is not
                    an obvious way to get a full list of ids from most ASE dbs besides simply looping
                    through the entire dataset. See the AseLMDBDataset which was written with this usecase
                    in mind.

            connect_args (dict): Keyword arguments for ase.db.connect()

            select_args (dict): Keyword arguments for ase.db.select()
                    You can use this to query/filter your database

            a2g_args (dict): Keyword arguments for ocpmodels.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True and/or "r_forces":True as appropriate
                    In that case, energy/forces must be in the database

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

            transform_args (dict): Additional keyword arguments for the transform callable

        atoms_transform (callable, optional): Additional preprocessing function applied to the Atoms
                    object. Useful for applying tags, for example.

        transform (callable, optional): Additional preprocessing function for the Data object
    """

    def load_dataset_get_ids(self, config) -> dummy_list:
        if isinstance(config["src"], list):
            filepaths = config["src"]
        elif os.path.isfile(config["src"]):
            filepaths = [config["src"]]
        elif os.path.isdir(config["src"]):
            filepaths = glob.glob(f'{config["src"]}/*')
        else:
            filepaths = glob.glob(config["src"])

        self.dbs = []

        for path in filepaths:
            try:
                self.dbs.append(
                    self.connect_db(path, config.get("connect_args", {}))
                )
            except ValueError:
                logging.warning(
                    f"Tried to connect to {path} but it's not an ASE database!"
                )

        self.select_args = config.get("select_args", {})
        if self.select_args is None:
            self.select_args = {}

        # In order to get all of the unique IDs using the default ASE db interface
        # we have to load all the data and check ids using a select. This is extremely
        # inefficient for large dataset. If the db we're using already presents a list of
        # ids and there is no query, we can just use that list instead and save ourselves
        # a lot of time!
        self.db_ids = []
        for db in self.dbs:
            if hasattr(db, "ids") and self.select_args == {}:
                self.db_ids.append(db.ids)
            else:
                self.db_ids.append(
                    [row.id for row in db.select(**self.select_args)]
                )

        idlens = [len(ids) for ids in self.db_ids]
        self._idlen_cumulative = np.cumsum(idlens).tolist()

        return dummy_list(sum(idlens))

    def get_atoms_object(self, idx):
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._idlen_cumulative, idx)

        # Extract index of element within that db
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._idlen_cumulative[db_idx - 1]
        assert el_idx >= 0

        atoms_row = self.dbs[db_idx]._get_row(self.db_ids[db_idx][el_idx])
        atoms = atoms_row.toatoms()

        if isinstance(atoms_row.data, dict):
            atoms.info.update(atoms_row.data)

        return atoms

    def connect_db(self, address, connect_args={}):
        if connect_args is None:
            connect_args = {}
        db_type = connect_args.get("type", "extract_from_name")
        if db_type == "lmdb" or (
            db_type == "extract_from_name" and address.split(".")[-1] == "lmdb"
        ):
            return LMDBDatabase(address, readonly=True, **connect_args)
        else:
            return ase.db.connect(address, **connect_args)

    def close_db(self) -> None:
        for db in self.dbs:
            if hasattr(db, "close"):
                db.close()

    def get_metadata(self):
        logging.warning(
            "You specific a folder of ASE dbs, so it's impossible to know which metadata to use. Using the first!"
        )
        if self.dbs[0].metadata == {}:
            return self.guess_target_metadata()
        else:
            return copy.deepcopy(self.dbs[0].metadata)

    def get_relaxed_energy(self, identifier):
        raise NotImplementedError(
            "IS2RE-Direct training with an ASE DB is not currently supported."
        )
