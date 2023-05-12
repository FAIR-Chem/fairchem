import pytest
from ase import build, db
from ase.io import write
import os

from ocpmodels.datasets import AseReadDataset, AseDBDataset

structures = [
    build.molecule("H2O"),
    build.bulk("Cu"),
    build.fcc111("Pt", size=[2,2,3], vacuum = 8, periodic=True),
]

def test_ase_read_dataset():
    for i, structure in enumerate(structures):
        write(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{i}.cif"), structure)
        
    dataset = AseReadDataset(config = {
        "src": os.path.join(os.path.dirname(os.path.abspath(__file__))),
        "suffix": ".cif"
    })
    
    assert len(dataset) == len(structures)
    
    for i in range(len(structures)):
        os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{i}.cif"))
    
def test_ase_db_dataset():
    try:
        os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db"))
    except FileNotFoundError:
        pass
    
    database = db.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db"))
    for i, structure in enumerate(structures):
        database.write(structure)
        
    dataset = AseDBDataset(config = {
        "src": os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db"),
    })
    
    assert len(dataset) == len(structures)
    
    os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "asedb.db"))