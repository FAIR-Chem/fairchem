from pathlib import Path
import ase
import warnings

from torch.utils.data import Dataset

from ocpmodels.common.registry import registry
from ocpmodels.preprocessing import AtomsToGraphs


@registry.register_dataset("ase_read")
class AseReadDataset(Dataset):
    """
    This Dataset uses ase.io.read to load data directly from a folder on disk.
    This is intended for small-scale testing and demonstrations of OCP.
    Larger datasets are better served by the effeciency of other dataset types
    such as LMDB.

    For a full list of ASE-readable filetypes, see
    https://wiki.fysik.dtu.dk/ase/ase/io/io.html

    args:
        config (dict):
            src (str): The source folder that contains your ASE-readable files

            suffix (str): The ending of the filepath of each file you want to read
                    ex. '/POSCAR', '.cif', '.xyz'

            a2g_config (dict): configuration for ocp.preprocessing.AtomsToGraphs
                    default options will work for most users

        transform (callable, optional): Additional preprocessing function
    """

    def __init__(self, config, transform=None):
        super(AseReadDataset, self).__init__()
        self.config = config

        self.path = Path(self.config["src"])
        if self.path.is_file():
            raise Exception("The specified src is not a directory")
        self.id = sorted(self.path.glob(f'*{self.config["suffix"]}'))

        a2g_config = config.get("a2g_config", {})
        self.a2g = AtomsToGraphs(
            max_neigh=a2g_config.get("max_neigh", 1000),
            radius=a2g_config.get("radius", 8),
            r_edges=a2g_config.get("r_edges", True),
            r_energy=a2g_config.get("r_energy", True),
            r_forces=a2g_config.get("r_forces", True),
            r_distances=a2g_config.get("r_distances", True),
            r_fixed=a2g_config.get("r_fixed", True),
            r_pbc=a2g_config.get("r_pbc", True),
        )

        self.transform = transform

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        try:
            atoms = ase.io.read(self.id[idx])
        except Exception as err:
            warnings.warn(f"{err} occured for: {self.id[idx]}")

        data_object = self.a2g.convert(atoms)

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object
