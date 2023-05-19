from pathlib import Path
import ase
import warnings
import numpy as np

from torch.utils.data import Dataset

from ocpmodels.common.registry import registry
from ocpmodels.preprocessing import AtomsToGraphs
from lmdb_database import LMDBDatabase


@registry.register_dataset("ase_read")
class AseReadDataset(Dataset):
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

            apply_tags (bool, optional): Apply a tag of one to each atom, which is
                    required of some models. A value of None will only tag structures
                    that are not already tagged.

        transform (callable, optional): Additional preprocessing function
    """

    def __init__(self, config, transform=None):
        super(AseReadDataset, self).__init__()
        self.config = config

        self.ase_read_args = config.get("ase_read_args", {})

        if ":" in self.ase_read_args.get("index", ""):
            raise NotImplementedError(
                "Multiple structures from one file is not currently supported"
            )

        self.path = Path(self.config["src"])
        if self.path.is_file():
            raise Exception("The specified src is not a directory")
        self.id = sorted(self.path.glob(f'{self.config["pattern"]}'))

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

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        try:
            atoms = ase.io.read(self.id[idx], **self.ase_read_args)
        except Exception as err:
            warnings.warn(f"{err} occured for: {self.id[idx]}")

        if self.config.get("apply_tags") == False:
            pass
        elif self.config.get("apply_tags") == True:
            atoms = self.apply_tags(atoms)
        elif sum(atoms.get_tags()) < 1:
            atoms = self.apply_tags(atoms)

        data_object = self.a2g.convert(atoms)

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    # This method is sometimes called by a trainer,
    # but there is nothing necessary to do here
    def close_db(self):
        pass

    def apply_tags(self, atoms):
        atoms.set_tags(np.ones(len(atoms)))
        return atoms


@registry.register_dataset("ase_db")
class AseDBDataset(Dataset):
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

            apply_tags (bool, optional): Apply a tag of 1 to each atom, which is
                    required of some models. A value of None will only tag structures
                    that are not already tagged.

        transform (callable, optional): Additional preprocessing function
    """

    def __init__(self, config, transform=None):
        super(AseDBDataset, self).__init__()
        self.config = config

        self.db = self.connect_db(
            self.config["src"], **self.config.get("connect_args", {})
        )

        self.select_args = self.config.get("select_args", {})

        self.id = [row.id for row in self.db.select(**self.select_args)]

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

    def __getitem__(self, idx):
        atoms = self.db.get_atoms(self.id[idx])

        if self.config.get("apply_tags") == False:
            pass
        elif self.config.get("apply_tags") == True:
            atoms = self.apply_tags(atoms)
        elif sum(atoms.get_tags()) < 1:
            atoms = self.apply_tags(atoms)

        data_object = self.a2g.convert(atoms)

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def __len__(self):
        return len(self.id)

    def connect_db(self, address, connect_args={}):
        if address[-5:] == ".lmdb":
            return LMDBDatabase(address, readonly=True, **connect_args)
        else:
            return ase.db.connect(address, **connect_args)

    def close_db(self):
        pass

    def apply_tags(self, atoms):
        atoms.set_tags(np.ones(len(atoms)))
        return atoms
