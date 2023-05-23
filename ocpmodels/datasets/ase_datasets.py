from pathlib import Path
import ase
import warnings
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm

from ocpmodels.common.registry import registry
from ocpmodels.preprocessing import AtomsToGraphs


class AseAtomsDataset(Dataset):
    """
    This is a base Dataset that includes helpful utilities for turning
    ASE atoms objects into OCP-usable data objects.

    Derived classes must add at least two things:
        self.get_atoms_object(): a function that takes an identifier and returns a corresponding atoms object
        self.id: a list of all possible identifiers that can be passed into self.get_atoms_object()
    """

    def __init__(self, config, transform=None):
        self.config = config

        a2g_args = config.get("a2g_args", {})
        self.a2g = AtomsToGraphs(
            max_neigh=a2g_args.get("max_neigh", 1000),
            radius=a2g_args.get("radius", 8),
            r_edges=a2g_args.get("r_edges", False),
            r_energy=a2g_args.get("r_energy", False),
            r_forces=a2g_args.get("r_forces", False),
            r_distances=a2g_args.get("r_distances", False),
            r_fixed=a2g_args.get("r_fixed", True),
            r_pbc=a2g_args.get("r_pbc", True),
        )

        self.transform = transform

        if self.config.get("keep_in_memory", False):
            self.data_objects = {}

        # Derived classes should extend this functionality to also create self.id,
        # a list of identifiers that can be passed to get_atoms_object()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        if self.config.get("keep_in_memory", False):
            if self.id[idx] in self.data_objects:
                return self.data_objects[self.id[idx]]

        atoms = self.get_atoms_object(self.id[idx])

        if self.config.get("apply_tags") == False:
            pass
        elif self.config.get("apply_tags") == True:
            atoms = self.apply_tags(atoms)
        elif np.all(atoms.get_tags() == 0):
            atoms = self.apply_tags(atoms)

        data_object = self.a2g.convert(atoms)

        if self.transform is not None:
            data_object = self.transform(data_object)

        if self.config.get("keep_in_memory", False):
            self.data_objects[self.id[idx]] = data_object

        return data_object

    def get_atoms_object(self, identifier):
        raise NotImplementedError
        # Derived classes should implement this method

    def close_db(self):
        pass
        # This method is sometimes called by a trainer

    def apply_tags(self, atoms):
        atoms.set_tags(np.ones(len(atoms)))
        return atoms
        # Derived classes may change this method for more complex tagging behavior


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

            a2g_args (dict): configuration for ocp.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True and/or "r_forces":True as appropriate
                    In that case, energy/forces must be in the files you read (ex. OUTCAR)

            ase_read_args (dict): additional arguments for ase.io.read()

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            apply_tags (bool, optional): Apply a tag of one to each atom, which is
                    required of some models. A value of None will only tag structures
                    that are not already tagged.

        transform (callable, optional): Additional preprocessing function
    """

    def __init__(self, config, transform=None):
        super(AseReadDataset, self).__init__(config, transform)
        self.ase_read_args = config.get("ase_read_args", {})

        if ":" in self.ase_read_args.get("index", ""):
            raise NotImplementedError(
                "To read multiple structures from a single file, please use AseReadMultiStructureDataset."
            )

        self.path = Path(self.config["src"])
        if self.path.is_file():
            raise Exception("The specified src is not a directory")
        self.id = sorted(self.path.glob(f'{self.config["pattern"]}'))

    def get_atoms_object(self, identifier):
        try:
            atoms = ase.io.read(identifier, **self.ase_read_args)
        except Exception as err:
            warnings.warn(f"{err} occured for: {identifier}")

        return atoms


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

            a2g_args (dict): configuration for ocp.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True and/or "r_forces":True as appropriate
                    In that case, energy/forces must be in the files you read (ex. OUTCAR)

            ase_read_args (dict): additional arguments for ase.io.read()

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            use_tqdm (bool): Use TQDM progress bar when initializing dataset

            apply_tags (bool, optional): Apply a tag of one to each atom, which is
                    required of some models. A value of None will only tag structures
                    that are not already tagged.

        transform (callable, optional): Additional preprocessing function
    """

    def __init__(self, config, transform=None):
        super(AseReadMultiStructureDataset, self).__init__(config, transform)
        self.ase_read_args = self.config.get("ase_read_args", {})
        if not hasattr(self.ase_read_args, "index"):
            self.ase_read_args["index"] = ":"

        self.path = Path(self.config["src"])
        if self.path.is_file():
            raise Exception("The specified src is not a directory")
        self.filenames = sorted(self.path.glob(f'{self.config["pattern"]}'))

        self.id = []
        if self.config.get("use_tqdm", True):
            self.filenames = tqdm(self.filenames)
        for filename in self.filenames:
            try:
                structures = ase.io.read(filename, **self.ase_read_args)
            except Exception as err:
                warnings.warn(f"{err} occured for: {filename}")
            else:
                for i, structure in enumerate(structures):
                    self.id.append(f"{filename} {i}")

                    if self.config.get("keep_in_memory", False):
                        data_object = self.a2g(structure)
                        if self.transform:
                            data_object = self.tranform(data_object)
                        self.data_objects[f"{filename} {i}"] = data_object

    def get_atoms_object(self, identifier):
        try:
            atoms = ase.io.read(
                "".join(identifier.split(" ")[:-1]), **self.ase_read_args
            )[int(identifier.split(" ")[-1])]
        except Exception as err:
            warnings.warn(f"{err} occured for: {identifier}")

        return atoms


@registry.register_dataset("ase_db")
class AseDBDataset(AseAtomsDataset):
    """
    This Dataset connects to an ASE Database, allowing the storage of atoms objects
    with a variety of backends including JSON, SQLite, and database server options.

    For more information, see:
    https://databases.fysik.dtu.dk/ase/ase/db/db.html

    args:
        config (dict):
            src (str): The path to or connection address of your ASE DB

            connect_args (dict): Additional arguments for ase.db.connect()

            select_args (dict): Additional arguments for ase.db.select()
                    You can use this to query/filter your database

            a2g_args (dict): configuration for ocp.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True and/or "r_forces":True as appropriate
                    In that case, energy/forces must be in the database

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            apply_tags (bool, optional): Apply a tag of 1 to each atom, which is
                    required of some models. A value of None will only tag structures
                    that are not already tagged.

        transform (callable, optional): Additional preprocessing function
    """

    def __init__(self, config, transform=None):
        super(AseDBDataset, self).__init__(config, transform)

        self.db = self.connect_db(
            self.config["src"], self.config.get("connect_args", {})
        )

        self.select_args = self.config.get("select_args", {})

        self.id = [row.id for row in self.db.select(**self.select_args)]

    def get_atoms_object(self, identifier):
        atoms = self.db._get_row(identifier).toatoms()
        return atoms

    def connect_db(self, address, connect_args={}):
        db_type = connect_args.get("type", "extract_from_name")
        if db_type == "lmdb" or (
            db_type == "extract_from_name" and address.split(".")[-1] == "lmdb"
        ):
            from ocpmodels.datasets.lmdb_database import LMDBDatabase

            return LMDBDatabase(address, readonly=True, **connect_args)
        else:
            return ase.db.connect(address, **connect_args)

    def close_db(self):
        if hasattr(self.db, "close"):
            self.db.close()
