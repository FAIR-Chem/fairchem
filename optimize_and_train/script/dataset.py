# File dataset.py save later 


import mongo
import os
import shutil
import numpy as np
import pandas as pd


def prepare_root_dir(main_dir):
    """
    make main_dir if none exists, 
    and prepare an empty root_dir for CgcnnDataset
    """
    # main_dir
    if not os.path.isdir(main_dir):
        os.mkdir(main_dir)
    # root_dir
    root_dir = os.path.join(main_dir, 'root_dir')
    if os.path.isdir(root_dir):
        shutil.rmtree(root_dir)
        os.mkdir(root_dir)
    else:
        os.mkdir(root_dir)


class CgcnnDatabaseSE:
    """make id.cif and id_prop.csv required for cgcnn in root_dir"""
    def __init__(self, doc, root_dir):
        self.doc = doc
        self.id = str(doc['mongo_id']) 
        self.atom_init = mongo.make_atoms_from_doc(doc['initial_configuration'])
        self.root_dir = root_dir

    def define_prop(self):
        """prop as surface energy, expected to be changed"""
        self.prop = self.doc['intercept']

    def create_cif(self):
        """write doc initial configuration into cif file"""
        assert os.path.exists(self.root_dir), 'root_dir does not exist!'
        cif_path = os.path.join(self.root_dir, '%s.cif'%self.id)
        self.atom_init.write(cif_path)

        
class CgcnnDatasetLogSE(CgcnnDatabaseSE):
    """customize CgcnnDatasetSE"""
    def __init__(self, doc, root_dir):
        CgcnnDatabaseSE.__init__(self, doc, root_dir)
    
    def define_prop(self):
        """prop as log(se) as new prop"""
        se_log = np.log(self.doc['intercept'])
        self.prop = se_log


class CgcnnDatasetDepth:
    def __init__(self, doc, root_dir):
        """make id_prop.cif from for atoms with few layers"""
        self.doc = doc
        self.id = str(doc['_id'])
        self.atom_init = mongo.make_atoms_from_doc(doc['initial_configuration'])
        self.fwid = doc['fwid']
        self.root_dir = root_dir
    
    def create_cif(self, bulk_id):
        """create cif files from client.atom from mongodb"""
        assert os.path.exists(self.root_dir)
        filename = 'bulk_' + bulk_id + '_depth_' + self.id + '_fwid_' + str(self.fwid)
        cif_path = os.path.join(self.root_dir, '%s.cif'%filename)
        self.atom_init.write(cif_path)
        self.filename = filename



if __name__ == '__main__':
    """self-test for generating database"""
    # prepare directory
    gga = 'ps'
    prop = 'se'
    selection = 'random'    # or selectmpid
    main_dir = '/home/wzhong1/cgcnn/new_0815/surf_en_filter/main_%s_%s_%s' % (gga, prop, selection)
    root_dir = os.path.join(main_dir, 'root_dir')
    prepare_root_dir(main_dir)

    # generating cif files for randomized docs and class object
    # LATER: PASS CLASS AS FUNCTION ARGUEMENT ! THEN WRITE IT AS FUNCTION IN CGCNNDATASET.PY
    random.shuffle(docs)
    se = []
    for doc in docs:
        x = CgcnnDatabaseSE(doc, root_dir)
        x.create_cif()
        x.define_prop()
        se.append([x.id, x.prop])

    # generating csv files
    pd.DataFrame(data=se).to_csv(os.path.join(root_dir, 'id_prop.csv'), index=False, header=False, float_format='%1.6f')
    print('%d objects generated' % len(se))
