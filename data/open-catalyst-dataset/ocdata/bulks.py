'''
This submodule contains the scripts that the Ulissi group used to pull the
relaxed bulk structures from our database.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

from tqdm import tqdm
import ase.db
from gaspy.gasdb import get_mongo_collection
from gaspy.mongo import make_atoms_from_doc


def dump_bulks(output_file_name='bulks.db'):
    with get_mongo_collection('atoms') as collection:
        docs = list(tqdm(collection.find({'fwname.calculation_type': 'unit cell optimization'}),
                         desc='pulling from FireWorks'))

    db = ase.db.connect(output_file_name)
    for doc in tqdm(docs, desc='writing to database'):
        atoms = make_atoms_from_doc(doc)
        n_elements = len(set(atoms.symbols))
        _ = db.write(atoms, mpid=doc['fwname']['mpid'], n_elements=n_elements)
