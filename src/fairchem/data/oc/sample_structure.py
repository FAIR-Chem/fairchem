import argparse

from ocdata.structure_sampler import StructureSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Sample adsorbate and bulk surface(s)')

    parser.add_argument('--seed', type=int, default=None, help='Random seed for sampling')

    # input and output
    parser.add_argument('--bulk_db', type=str, required=True, help='Underlying db for bulks')
    parser.add_argument('--adsorbate_db', type=str, required=True, help='Underlying db for adsorbates')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory for outputs')

    # for optimized (automatically try to use optimized if this is provided)
    parser.add_argument('--precomputed_structures', type=str, default=None, help='Root directory of precomputed structures')

    # args for enumerating all combinations:
    parser.add_argument('--enumerate_all_structures', action='store_true', default=False,
        help='Find all possible structures given a specific adsorbate and a list of bulks')
    parser.add_argument('--adsorbate_index', type=int, default=None, help='Adsorbate index (int)')
    parser.add_argument('--bulk_indices', type=str, default=None, help='Comma separated list of bulk indices')
    parser.add_argument('--surface_index', type=int, default=None, help='Optional surface index (int)')

    parser.add_argument('--verbose', action='store_true', default=False, help='Log detailed info')

    # check that all needed args are supplied
    args = parser.parse_args()
    if args.enumerate_all_structures:
        if args.adsorbate_index is None or args.bulk_indices is None:
            parser.error('Enumerating all structures requires specified adsorbate and bulks')

    elif args.seed is None:
            parser.error('Seed is required when sampling one random structure')
    return args

if __name__ == '__main__':
    args = parse_args()
    job = StructureSampler(args)
    job.run()
