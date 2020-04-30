'''
Helper script convert db files to pkl files.
'''

import ase
import ase.db
import pickle


def convert_bulk(input_bulk_database, max_num_elements, output_pkl):
    '''
    Converts an input ASE.db to an inverted index to efficiently sample bulks
    '''
    assert max_num_elements > 0
    db = ase.db.connect(input_bulk_database)

    index = {}
    total_entries = 0
    for i in range(1, max_num_elements + 1):
        index[i] = []
        rows = list(db.select(n_elements=i))
        for r in range(len(rows)):
            index[i].append((rows[r].toatoms(), rows[r].mpid))
            total_entries += 1
    with open(output_pkl, 'wb') as f:
        pickle.dump(index, f)

    # As of bulk.db file from Kevin on April 29 2020
    assert total_entries == 16180


def convert_adsorbate(input_adsorbate_database, output_pkl):
    '''
    Converts an input ASE.db to an inverted index to efficiently sample adsorbates
    '''
    db = ase.db.connect(input_adsorbate_database)

    index = {}
   
    for i, row in enumerate(db.select()):
        atoms = row.toatoms()
        data = row.data
        smiles = data['SMILE']
        bond_indices = data['bond_idx']
        index[i] = (atoms, smiles, bond_indices)
       
    with open(output_pkl, 'wb') as f:
        pickle.dump(index, f)

    # As of adsorbates.db file in master on April 28 2020
    assert len(index) == 82


def main():
    #convert_bulk("ase_dbs/bulks.db", 3, "ase_pkl/bulks.pkl")
    convert_adsorbate("ase_dbs/adsorbates.db", "ase_pkl/adsorbates.pkl")


if __name__ == "__main__":
    main()
