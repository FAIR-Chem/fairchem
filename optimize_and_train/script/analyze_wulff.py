# file analyze_wulff.py
# load cgcnn results into dict and make wulff structure for each mpid 

import os
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.analysis import wulff
from pymatgen.ext.matproj import MPRester
from sklearn.metrics import mean_absolute_error, mean_squared_error
from copy import deepcopy
from collections import defaultdict
api_key = 'MGOdX3P4nI18eKvE'


def load_cgcnn_results(path, train_tag=0):
    """
    save data from cgcnn test_result.csv into dict with mongo_id as key
    """
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        results = {}
        for row in reader:
            results[row[0]] = {'target': float(row[1]), 
                               'predict': float(row[2]), 
                               'train_tag': train_tag}
            
    return results

def merge_cgcnn_results(path_test, path_all):
    """
    combine all and test rest_result.csv 
    and differentiate train and test set on train_tag
    """
    results_all = load_cgcnn_results(path_all, train_tag=1)
    results_test = load_cgcnn_results(path_test, train_tag=0)
    results = {}
    results = results_all.copy()
    results.update(results_test)
    return results



class ModifyDocs:
    """actions on doc"""
    def __init__(self, doc, cgcnn):
        self.mongo_id = str(doc['mongo_id'])
        self.mpid = doc['mpid']
        self.miller = doc['miller']
        self.intercept = doc['intercept']
        self.uncertainty = doc['intercept_uncertainty']
        self.symbol_counts = doc['atoms']['symbol_counts']
        self.formula = doc['formula']
        self.spacegroup = doc['atoms']['spacegroup']
        self.se_target = cgcnn[self.mongo_id]['target']
        self.se_predict = cgcnn[self.mongo_id]['predict']
        self.train_tag = cgcnn[self.mongo_id]['train_tag']
        

class ModifyMpids:
    """group mpids and make changes""" 
    def __init__(self, mpid, docs, cgcnn):
        objs = [ModifyDocs(doc, cgcnn) for doc in docs]
        self.mpid = mpid
        self.match_objs = [obj for obj in objs if obj.mpid == self.mpid]
        self.spacegroup = self.match_objs[0].spacegroup
        self.symbols = list(self.match_objs[0].symbol_counts.keys())
        
        # find subset with lowest miller energy
        match = sorted(self.match_objs, key = lambda x: getattr(x, 'miller'))
        match_subset = []
        m0 = match[0]
        for m in match:
            if m.miller == m0.miller:
                if m.intercept < m0.intercept:
                    m0 = deepcopy(m)
            else:
                match_subset.append(m0)
                m0 = deepcopy(m)
        match_subset.append(m0)
        self.miller_lowest = match_subset
        self.mpid_lowest = sorted(self.match_objs, key = lambda x: getattr(x, 'intercept'))[0]
        
    def take_tag(self):
        """take index of train data in miller lowest set"""
        train_tag = [m.train_tag for m in self.miller_lowest]
        idx_train = [i for i,j in enumerate(train_tag) if j == 1]
        idx_test = [i for i,j in enumerate(train_tag) if j == 0]
        return idx_train, idx_test
        
    def take_surface_energy(self):
        """ surface energy for miller lowest set"""
        se_target = [m.se_target for m in self.miller_lowest]
        se_predict = [m.se_predict for m in self.miller_lowest]
        return se_target, se_predict
        
    def take_ratio(self):
        """surfenergy energy ratio normalized by lowest surface energy for miller lowest subset"""
        ratio_target =  [m.se_target / self.mpid_lowest.se_target for m in self.miller_lowest]
        ratio_predict = [m.se_predict / self.mpid_lowest.se_predict for m in self.miller_lowest]
        return ratio_target, ratio_predict
        
    def wulff_shape(self):
        """construct wulff shape"""
        millers =[m.miller for m in self.miller_lowest]
        se_targets = [m.se_target for m in self.miller_lowest]
        se_predicts = [m.se_predict for m in self.miller_lowest]
        
        if len(millers) > 1:
            with MPRester(api_key) as m:
                structure = m.get_structure_by_material_id(self.mpid)
                ws_t = wulff.WulffShape(structure.lattice, millers, se_targets, symprec=1e-5)
                ws_p = wulff.WulffShape(structure.lattice, millers, se_predicts, symprec=1e-5)
        return ws_t, ws_p
        