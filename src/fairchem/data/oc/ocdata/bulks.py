'''
This submodule contains the scripts that the Ulissi group used to pull the
relaxed bulk structures from our database.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import warnings
from tqdm import tqdm
import ase.db
from gaspy.gasdb import get_mongo_collection
from gaspy.mongo import make_atoms_from_doc


with get_mongo_collection('atoms') as collection:
    docs = list(tqdm(collection.find({'fwname.calculation_type': 'unit cell optimization',
                                      'fwname.vasp_settings.gga': 'RP',
                                      'fwname.vasp_settings.pp': 'PBE',
                                      'fwname.vasp_settings.xc': {'$exists': False},
                                      'fwname.vasp_settings.pp_version': '5.4',
                                      'fwname.vasp_settings.encut': 500,
                                      'fwname.vasp_settings.isym': 0}),
                     desc='pulling from FireWorks'))

mpids = set()
db = ase.db.connect('bulks.db')
for doc in tqdm(docs, desc='writing to database'):
    atoms = make_atoms_from_doc(doc)
    n_elements = len(set(atoms.symbols))

    if n_elements <= 3:
        mpid = doc['fwname']['mpid']
        if mpid not in mpids:
            mpids.add(mpid)
            _ = db.write(atoms, mpid=doc['fwname']['mpid'], n_elements=n_elements)

        else:
            warnings.warn('Found a duplicate MPID:  %s; adding the first one' % mpid,
                          RuntimeWarning)
