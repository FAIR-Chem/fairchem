import pymongo
import mongo
import os
import csv
import random
import shutil
import numpy as np
import pandas as pd
import pprint

import matplotlib.pyplot as plt
from pymatgen.analysis import wulff
from pymatgen.ext.matproj import MPRester

# connect host and port, grab database form mongodb, and authenticate
MC = pymongo.MongoClient(host='localhost',port=27017)
database=MC.vasp_zu_vaspsurfaces
database.authenticate('admin_zu_vaspsurfaces','Ff300D7E8p')

def fingerprints_se():
    '''
    this modifies gaspy.defaults.fingerprints, and adds keys for new mongo docs.
    '''
    fingerprints = {'mongo_id': '$_id',
                    'mpid': '$processed_data.calculation_info.mpid',
                    'formula': '$processed_data.calculation_info.formula',
                    'miller': '$processed_data.calculation_info.miller',
                    'shift': '$processed_data.calculation_info.shift',
                    'intercept': '$processed_data.surface_energy_info.intercept',
                    'intercept_uncertainty': '$processed_data.surface_energy_info.intercept_uncertainty', 
                    'max_surface_movement': '$processed_data.movement_data.max_surface_movement', 
                    'atoms': '$atoms',
                    'initial_configuration': '$initial_configuration',
                    'FW_info': '$processed_data.FW_info'}
    return fingerprints


def get_docs_se(client=database, collection_name='surface_energy', fingerprints=None, 
                calc_settings='pbe-sol', vasp_settings=None, 
                intercept_min=None, intercept_max=None,
                intercept_uncertainty_min=None, intercept_uncertainty_max=None):
    '''
    this applies filters to clients.
    database is defined before as function similar to gaspy.gasdb.get_adsorption_client()
    '''
    # fingerprints for surface energy
    if not fingerprints:
        fingerprints = fingerprints_se()
    
    # put "fingerprinting" into a 'group' directory
    group = {'$group': {'_id': fingerprints}}
    match = {'$match': {}}
    
    # create match filters
    # calc_settings
    if not calc_settings:
        pass
    elif calc_settings == 'rpbe':
        match['$match']['processed_data.vasp_settings.gga'] = 'RP'
    elif calc_settings == 'beef-vdw':
        match['$match']['processed_data.vasp_settings.gga'] = 'BF'
    elif calc_settings == 'pbe-sol':
        match['$match']['processed_data.vasp_settings.gga'] = 'PS'
    else:
        raise Exception('Unknown calc_settings')
        
    # vasp_settings
    if vasp_settings:
        for key, value in vasp_settings.iteritems():
            match['$match']['processed_data.vasp_setting.%s' % key] = value
        if ('gga' in vasp_settings and calc_settings):
            warnings.warn('User specified both calc_settings and vasp_settings.gga. GASpy will default to the given vasp_settings.gga', SyntaxWarning)
    
    # intercept constraints
    if intercept_min == 0:
        intercept_min = 1e-20
        
    if (intercept_max and intercept_min):
        match['$match']['processed_data.surface_energy_info.intercept'] = {'$gt': intercept_min, '$lt': intercept_max}
    elif (intercept_max and not intercept_min):
        match['$match']['processed_data.surface_energy_info.intercept'] = {'$lt': intercept_max}
    elif (not intercept_max and intercept_min):
        match['$match']['processed_data.surface_energy_info.intercept'] = {'$gt': intercept_min}
    
    # intercept constraints
    if intercept_uncertainty_min == 0:
         intercept_uncertainty_min = 1e-20
        
    if (intercept_uncertainty_max and intercept_uncertainty_min):
        match['$match']['processed_data.surface_energy_info.intercept_uncertainty'] = {'$gt': intercept_uncertainty_min, '$lt': intercept_uncertainty_max}
    elif (intercept_uncertainty_max and not intercept_uncertainty_min):
        match['$match']['processed_data.surface_energy_info.intercept_uncertainty'] = {'$lt': intercept_uncertainty_max}
    elif (not intercept_uncertainty_max and intercept_uncertainty_min):
        match['$match']['processed_data.surface_energy_info.intercept_uncertainty'] = {'$gt': intercept_uncertainty_min}
    
    # compile pipeline; add matches only if any matches are specified 
    if match['$match']:
        pipeline = [match, group]
    else:
        pipline = [group]
    
    if collection_name == 'catalog':
        # what is similar to get_catalog_client_readonly().db using database
        # need to change here !
        # collection = getattr(get_catalog_client_readonly().db, collection_name)
        print('collection name is "catalog", not understand how to modify this')
    else:
        collection = getattr(client, collection_name)
    
    print('Starting to pull documnets ...')
    cursor = collection.aggregate(pipeline, allowDiskUse=True, useCursor=True)
    docs = [doc['_id'] for doc in cursor]
    
    if not docs:
        warnings.warn('We did not find any matching documents', RuntimeWarning)
    
    for doc in docs:
        if set(fingerprints.keys()).issubset(doc.keys()) and all(doc.values()):
            pass
        else:
            del doc
    
    return docs


def doc_filters_se():
    '''
    define filters for surface energy
    '''
    filters = dict(intercept_min = 0.0,
                   intercept_max = 0.3,
                   intercept_uncertainty_min = 0.0,
                   intercept_uncertainty_max = 0.01)
    return filters

def get_num_elements(docs, num_elements):
    '''
    select docs with specific list number of elements
    '''
    docs_elements = [doc for doc in docs if len(doc['atoms']['symbol_counts'].keys()) in num_elements]
    return docs_elements
