from pathlib import Path
import ase
import warnings
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm

from ocpmodels.common.registry import registry
from ocpmodels.preprocessing import AtomsToGraphs


def apply_one_tags(atoms, skip_if_nonzero=True, skip_always=False):
    """
    This function will apply tags of 1 to an ASE atoms object.
    It is used as an atoms_transform in the datasets contained in this file.

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


class AseAtomsDataset(Dataset):
    """
    This is a base Dataset that includes helpful utilities for turning
    ASE atoms objects into OCP-usable data objects.

    Derived classes must add at least two things:
        self.get_atoms_object(): a function that takes an identifier and returns a corresponding atoms object
        self.id: a list of all possible identifiers that can be passed into self.get_atoms_object()
    Identifiers need not be any particular type.
    """

    def __init__(self, config, transform=None, atoms_transform=apply_one_tags):
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
        self.atoms_transform = atoms_transform

        if self.config.get("keep_in_memory", False):
            self.data_objects = {}

        # Derived classes should extend this functionality to also create self.id,
        # a list of identifiers that can be passed to get_atoms_object()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        # Handle slicing
        if isinstance(idx, slice):
            return [self[i] for i in range(len(self.id))[idx]]

        # Check if data object is already in memory
        if self.config.get("keep_in_memory", False):
            if self.id[idx] in self.data_objects:
                return self.data_objects[self.id[idx]]

        # Get atoms object via derived class method
        atoms = self.get_atoms_object(self.id[idx])

        # Transform atoms object
        if self.atoms_transform is not None:
            atoms = self.atoms_transform(
                atoms, **self.config.get("atoms_transform_args", {})
            )

        # Convert to data object
        data_object = self.a2g.convert(atoms)

        # Transform data object
        if self.transform is not None:
            data_object = self.transform(
                data_object, **self.config.get("transform_args", {})
            )

        # Save in memory, if specified
        if self.config.get("keep_in_memory", False):
            self.data_objects[self.id[idx]] = data_object

        return data_object

    def get_atoms_object(self, identifier):
        raise NotImplementedError
        # Derived classes should implement this method

    def close_db(self):
        pass
        # This method is sometimes called by a trainer


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

            a2g_args (dict): Keyword arguments for ocp.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True and/or "r_forces":True as appropriate
                    In that case, energy/forces must be in the files you read (ex. OUTCAR)

            ase_read_args (dict): Keyword arguments for ase.io.read()

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

            transform_args (dict): Additional keyword arguments for the transform callable

        atoms_transform (callable, optional): Additional preprocessing function applied to the Atoms
                    object. Useful for applying tags.

        transform (callable, optional): Additional preprocessing function for the Data object

    """

    def __init__(self, config, transform=None, atoms_transform=apply_one_tags):
        super(AseReadDataset, self).__init__(
            config, transform, atoms_transform
        )
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
            raise err

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

            a2g_args (dict): Keyword arguments for ocp.preprocessing.AtomsToGraphs()
                    default options will work for most users

                    If you are using this for a training dataset, set
                    "r_energy":True and/or "r_forces":True as appropriate
                    In that case, energy/forces must be in the files you read (ex. OUTCAR)

            ase_read_args (dict): Keyword arguments for ase.io.read()

            keep_in_memory (bool): Store data in memory. This helps avoid random reads if you need
                    to iterate over a dataset many times (e.g. training for many epochs).
                    Not recommended for large datasets.

            use_tqdm (bool): Use TQDM progress bar when initializing dataset

            atoms_transform_args (dict): Additional keyword arguments for the atoms_transform callable

            transform_args (dict): Additional keyword arguments for the transform callable

        atoms_transform (callable, optional): Additional preprocessing function applied to the Atoms
            object. Useful for applying tags.

        transform (callable, optional): Additional preprocessing function for the Data object
    """

    def __init__(self, config, transform=None, atoms_transform=apply_one_tags):
        super(AseReadMultiStructureDataset, self).__init__(
            config, transform, atoms_transform
        )
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
                        # Transform atoms object
                        if self.atoms_transform is not None:
                            atoms = self.atoms_transform(
                                structure,
                                **self.config.get("atoms_transform_args", {}),
                            )

                        # Convert to data object
                        data_object = self.a2g.convert(atoms)

                        # Transform data object
                        if self.transform is not None:
                            data_object = self.transform(
                                data_object,
                                **self.config.get("transform_args", {}),
                            )

                        self.data_objects[f"{filename} {i}"] = data_object

    def get_atoms_object(self, identifier):
        try:
            atoms = ase.io.read(
                "".join(identifier.split(" ")[:-1]), **self.ase_read_args
            )[int(identifier.split(" ")[-1])]
        except Exception as err:
            warnings.warn(f"{err} occured for: {identifier}")
            raise err

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

            connect_args (dict): Keyword arguments for ase.db.connect()

            select_args (dict): Keyword arguments for ase.db.select()
                    You can use this to query/filter your database

            a2g_args (dict): Keyword arguments for ocp.preprocessing.AtomsToGraphs()
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
                    object. Useful for applying tags.

        transform (callable, optional): Additional preprocessing function for the Data object
    """

    def __init__(self, config, transform=None, atoms_transform=apply_one_tags):
        super(AseDBDataset, self).__init__(config, transform, atoms_transform)

        self.db = self.connect_db(
            self.config["src"], self.config.get("connect_args", {})
        )

        self.select_args = self.config.get("select_args", {})

        self.id = [row.id for row in self.db.select(**self.select_args)]

    def get_atoms_object(self, identifier):
        return self.db._get_row(identifier).toatoms()

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
