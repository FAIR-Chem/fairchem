
from ocdata.vasp import run_vasp, write_vasp_input_files
from ocdata.adsorbates import Adsorbate
from ocdata.bulk_obj import Bulk
from ocdata.surfaces import Surface
from ocdata.combined import Combined

import argparse
import logging
import math
import numpy as np
import os
import pickle
import time


class StructureSampler():
    '''
    Writes vasp input files for one of the following:
    - one adsorbate/bulk/surface/config, based on a random seed
    - one specified adsorbate, n specified bulks, and all possible surfaces and configs

    The output directory structure will look like the following: for sampling a random structure,
    the directories will be `random{seed}_surface` and `random{seed}_adslab` for the surface alone
    and the adsorbate+surface, respectively. For enumerating all structures, the directories will be
    `{adsorbate}_{bulk}_{surface}_surface` and `{adsorbate}_{bulk}_{surface}_adslab{config}`, where
    everything in braces are the respective indices.

    '''
    def __init__(self, args, adsorbate_index=None, bulk_indices_list=None):
        # set up args, random seed, and logging
        self.args = args

        self.logger = logging.getLogger()
        logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S')
        self.logger.setLevel(logging.INFO if self.args.verbose else logging.WARNING)

        if adsorbate_index is not None and bulk_indices_list is not None:
            self.adsorbate_index = adsorbate_index
            self.bulk_indices_list = bulk_indices_list
            self.logger.info(f'Enumerating all surfaces/configs for adsorbate {adsorbate_index} and bulks {bulk_indices_list}')
        else:
            self.logger.info('Sampling one random structure')

        np.random.seed(self.args.seed)

    def run(self):
        start = time.time()

        if self.args.enumerate_all_structures:
            self.adsorbate = Adsorbate(self.args.adsorbate_db, self.adsorbate_index)
        self.load_bulks()
        self.load_and_write_surfaces()

        end = time.time()
        print(f'Done! ({round(end - start, 2)}s)')

    def load_bulks(self):
        '''
        Loads bulk structures (one random or a list of specified ones)
        and stores them in self.all_bulks
        '''
        self.all_bulks = []
        with open(self.args.bulk_db, 'rb') as f:
            inv_index = pickle.load(f)

        if self.args.enumerate_all_structures:
            for ind in self.bulk_indices_list:
                self.all_bulks.append(Bulk(inv_index, self.args.precomputed_structures, int(ind)))
        else:
            self.all_bulks.append(Bulk(inv_index, self.args.precomputed_structures))

    def load_and_write_surfaces(self):
        '''
        Loops through all bulks and chooses one random or all possible surfaces;
        writes info for that surface and combined surface+adsorbate
        '''
        for bulk_ind, bulk in enumerate(self.all_bulks):
            possible_surfaces = bulk.get_possible_surfaces()
            if self.args.enumerate_all_structures:
                self.logger.info(f'Enumerating all {len(possible_surfaces)} surfaces for bulk {self.bulk_indices_list[bulk_ind]}')
                for surface_ind, surface_info in enumerate(possible_surfaces):
                    surface = Surface(bulk, surface_info, surface_ind, len(possible_surfaces))
                    self.combine_and_write(surface, self.bulk_indices_list[bulk_ind], surface_ind)
            else:
                surface_info_index = np.random.choice(len(possible_surfaces))
                surface = Surface(bulk, possible_surfaces[surface_info_index], surface_info_index, len(possible_surfaces))
                self.adsorbate = Adsorbate(self.args.adsorbate_db)
                self.combine_and_write(surface)

    def combine_and_write(self, surface, cur_bulk_index=None, cur_surface_index=None):
        '''
        Add the adsorbate onto a given surface in a Combined object.
        Writes output files for the surface itself and the combined surface+adsorbate
        '''
        if self.args.enumerate_all_structures:
            output_name_template = f'{self.adsorbate_index}_{cur_bulk_index}_{cur_surface_index}'
        else:
            output_name_template = f'random{self.args.seed}'

        self.write_surface(surface, output_name_template)

        combined = Combined(self.adsorbate, surface, self.args.enumerate_all_structures)
        self.write_adsorbed_surface(combined, output_name_template)

    def write_surface(self, surface, output_name_template):
        # write files for just the surface
        bulk_dict = surface.get_bulk_dict()
        bulk_dir = os.path.join(self.args.output_dir, output_name_template + '_surface')
        write_vasp_input_files(bulk_dict['bulk_atomsobject'], bulk_dir)
        self.write_metadata_pkl(bulk_dict, os.path.join(bulk_dir, 'metadata.pkl'))
        self.logger.info(f"wrote surface ({bulk_dict['bulk_samplingstr']}) to {bulk_dir}")

    def write_adsorbed_surface(self, combined, output_name_template):
        # write files for adsorbate placed on surface
        self.logger.info(f'Writing {combined.num_configs} adslab configs')
        for config_ind in range(combined.num_configs):
            if self.args.enumerate_all_structures:
                adsorbed_bulk_dir = os.path.join(self.args.output_dir, output_name_template + f'_adslab{config_ind}')
            else:
                adsorbed_bulk_dir = os.path.join(self.args.output_dir, output_name_template + '_adslab')
            adsorbed_bulk_dict = combined.get_adsorbed_bulk_dict(config_ind)
            write_vasp_input_files(adsorbed_bulk_dict['adsorbed_bulk_atomsobject'], adsorbed_bulk_dir)
            self.write_metadata_pkl(adsorbed_bulk_dict, os.path.join(adsorbed_bulk_dir, 'metadata.pkl'))
            self.logger.info(f"wrote adsorbed surface ({adsorbed_bulk_dict['adsorbed_bulk_samplingstr']}) to {adsorbed_bulk_dir}")

    def write_metadata_pkl(self, dict_to_write, path):
        file_path = os.path.join(path, 'metadata.pkl')
        with open(path, 'wb') as f:
            pickle.dump(dict_to_write, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Sample adsorbate and bulk surface(s)')

    parser.add_argument('--seed', type=int, default=None, help='Random seed for sampling')

    # input and output
    parser.add_argument('--bulk_db', type=str, required=True, help='Underlying db for bulks')
    parser.add_argument('--adsorbate_db', type=str, required=True, help='Underlying db for adsorbates')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir path')

    # for optimized (automatically try to use optimized if this is provided)
    parser.add_argument('--precomputed_structures', type=str, default=None, help='Root directory of precomputed structures')

    # args for enumerating all combinations:
    parser.add_argument('--enumerate_all_structures', action='store_true', default=False,
        help='Find all possible structures given a specific adsorbate and a list of bulks')
    parser.add_argument('--adsorbate_index', type=int, default=None, help='adsorbate index (int)')
    parser.add_argument('--bulk_indices', type=str, default=None, help='Comma separated list of bulk indices')

    # read adsorbate and bulk indices from file:
    parser.add_argument('--indices_file', type=str, default=None, help='Pickle containing tuples of (bulk, adsorbate) strings')
    parser.add_argument('--adsorbate_index_mapping', type=str, default=None, help='Text file that maps index to adsorbate')
    parser.add_argument('--mpid_index_mapping', type=str, default=None, help='Text file that maps index to mpid')

    parser.add_argument('--verbose', action='store_true', default=False, help='Log detailed info')

    # check that all needed args are supplied
    args = parser.parse_args()
    if args.enumerate_all_structures:
        if args.indices_file:
            if args.adsorbate_index_mapping is None or args.mpid_index_mapping is None:
                parser.error('Must provide mapping of index to mpids and index to adsorbate strings')
        elif args.adsorbate_index is None or args.bulk_indices is None:
            parser.error('Enumerating all structures requires specified adsorbate and bulks')

    elif args.seed is None:
            parser.error('Seed is required when sampling one random structure')
    return args

def invert_mappings(args):
    # takes two files that map index to mpid or smiles, and returns the inverse dict
    # that maps mpid/smiles to bulk/adsorbate indices
    mpid_to_ind = {}
    with open(args.mpid_index_mapping, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elems = line.strip().split(' ')
            assert len(elems) == 3
            mpid_to_ind[elems[1]] = (int(elems[0]), elems[2])

    smiles_to_ind = {}
    with open(args.adsorbate_index_mapping, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elems = line.strip().split(' ')
            assert len(elems) == 2
            smiles_to_ind[elems[1]] = int(elems[0])

    return mpid_to_ind, smiles_to_ind

if __name__ == '__main__':
    args = parse_args()

    if args.indices_file:
        # enumerate all structures for all given (adsorbate, bulk) tuples
        mpid_to_ind, smiles_to_ind = invert_mappings(args)
        with open(args.indices_file, 'rb') as f:
            pairs = pickle.load(f)
        for mpid, smiles in pairs:
            job = StructureSampler(args, smiles_to_ind[smiles], [mpid_to_ind[mpid][0]])
            job.run()

    elif args.enumerate_all_structures:
        # enumerate all structures for one adsorbate and a list of bulks
        job = StructureSampler(args, args.adsorbate_index, args.bulk_indices.split(','))
        job.run()
    else:
        # sample one random adsorbate, bulk, surface, and config
        job = StructureSampler(args)
        job.run()
