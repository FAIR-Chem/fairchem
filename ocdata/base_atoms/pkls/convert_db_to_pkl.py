'''
Helper script convert db files to pkl files.
'''

__author__ = 'Siddharth Goyal'

import ase
import ase.db
import pickle


def get_bulk_inverted_index_1(input_bulk_database, max_num_elements):
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
        print(len(rows))
        for r in range(len(rows)):
            index[i].append((rows[r].toatoms(), rows[r].mpid))
            total_entries += 1

    return index, total_entries

def get_bulk_inverted_index_2(input_bulk_database, max_num_elements):
    '''
    Converts an input ASE.db to an inverted index to efficiently sample bulks
    '''
    assert max_num_elements > 0
    db = ase.db.connect(input_bulk_database)
    rows = list(db.select())

    index = {}
    total_entries = 0
    for r in range(len(rows)):
        bulk = rows[r].toatoms()
        mpid = rows[r].mpid
        formula_str = str(bulk.symbols)
        num_ele = sum(1 for c in formula_str if c.isupper())
        if num_ele > max_num_elements:
            continue
        if num_ele not in index:
            index[num_ele] = []
        index[num_ele].append((bulk, mpid))
        total_entries += 1

    return index, total_entries


# handling 2 dbs 
def convert_bulk(bulk_path1, bulk_path2, max_num_elements, output_pkl, precompute_pkl_for_surface_enumeration):

    index1, total_entries1 = get_bulk_inverted_index_1(bulk_path1, max_num_elements)
    index2, total_entries2 = get_bulk_inverted_index_2(bulk_path2, max_num_elements)

    # As of bulk.db file from Kevin on 01 May 2020
    assert total_entries1 == 11010
    assert total_entries2 == 491
    
    combined_total_entries = total_entries1 + total_entries2
    lst_for_surface_enumeration = []
    combined_index = {}
    all_index_counter = 0

    # Handle first db elements
    for i in range(1, max_num_elements + 1):
        combined_index[i] = []

        for j in range(len(index1[i])):
            sampling_str = str(j) + "/" + str(len(index1[i])) + "_" + str(all_index_counter) + "/11010"
            bulk, mpid = index1[i][j]
            current_obj = (bulk, mpid, sampling_str, all_index_counter)
            print(current_obj)
            combined_index[i].append(current_obj)
            all_index_counter += 1
            lst_for_surface_enumeration.append(current_obj)

    # Handle second db elements
    for i in range(1, max_num_elements + 1):
        for j in range(len(index2[i])):
            sampling_str = str(j + len(index1[i])) + "/"  + str(len(index1[i]) + len(index2[i])) + "_" + str(all_index_counter) + "/" + str(combined_total_entries)
            bulk, mpid = index2[i][j]
            current_obj = (bulk, mpid, sampling_str, all_index_counter)
            print(current_obj)
            combined_index[i].append(current_obj)
            all_index_counter += 1
            lst_for_surface_enumeration.append(current_obj)

    with open(output_pkl, 'wb') as f:
        pickle.dump(combined_index, f)

    with open(precompute_pkl_for_surface_enumeration, 'wb') as g:
        pickle.dump(lst_for_surface_enumeration, g)
    

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
    convert_bulk("../ase_dbs/bulks.db", "../ase_dbs/new_bulks.db", 3, "bulks_may12.pkl", "for_surface_enumeration_bulk_may12.pkl")
#    convert_adsorbate("../ase_dbs/adsorbates.db", "adsorbates.pkl")


if __name__ == "__main__":
    main()
