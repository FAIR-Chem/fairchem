import argparse
import logging
import multiprocessing as mp
import os
import pickle
import time
import traceback

import numpy as np
from tqdm import tqdm

from ocdata.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
from ocdata.utils.vasp import write_vasp_input_files


class StructureGenerator:
    """
    A class that creates adsorbate/bulk/slab objects given specified indices,
    and writes vasp input files and metadata for multiple placements of the adsorbate
    on the slab. You can choose random, heuristic, or both types of placements.

    The output directory structure will have the following nested structure,
    where "files" represents the vasp input files and the metadata.pkl:
        outputdir/
            bulk0/
                surface0/
                    surface/files
                    ads0/
                        heur0/files
                        heur1/files
                        rand0/files
                        ...
                    ads1/
                        ...
                surface1/
                    ...
            bulk1/
                ...

    Precomputed surfaces will be calculated and saved out if they don't
    already exist in the provided directory.

    Arguments
    ----------
    args: argparse.Namespace
        Contains all command line args
    bulk_index: int
        Index of the bulk within the bulk db
    surface_index: int
        Index of the surface in the list of all possible surfaces
    adsorbate_index: int
        Index of the adsorbate within the adsorbate db
    """

    def __init__(self, args, bulk_index, surface_index, adsorbate_index):
        """
        Set up args from argparse, random seed, and logging.
        """
        self.args = args
        self.bulk_index = bulk_index
        self.surface_index = surface_index
        self.adsorbate_index = adsorbate_index

        self.logger = logging.getLogger()
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        self.logger.setLevel(logging.INFO if self.args.verbose else logging.WARNING)

        self.logger.info(
            f"Starting adsorbate {self.adsorbate_index}, bulk {self.bulk_index}, surface {self.surface_index}"
        )
        if self.args.seed:
            np.random.seed(self.args.seed)

    def run(self):
        """
        Create adsorbate/bulk/surface objects, generate adslab placements,
        and write to files.
        """
        start = time.time()

        # create adsorbate, bulk, and surface objects
        self.bulk = Bulk(
            bulk_id_from_db=self.bulk_index, bulk_db_path=self.args.bulk_db
        )
        self.adsorbate = Adsorbate(
            adsorbate_id_from_db=self.adsorbate_index,
            adsorbate_db_path=self.args.adsorbate_db,
        )
        all_slabs = self.bulk.get_slabs(
            max_miller=self.args.max_miller,
            precomputed_slabs_dir=self.args.precomputed_slabs_dir,
        )
        self.slab = all_slabs[self.surface_index]

        # create adslabs
        self.rand_adslabs, self.heur_adslabs = None, None
        if self.args.heuristic_placements:
            self.heur_adslabs = AdsorbateSlabConfig(
                self.slab,
                self.adsorbate,
                num_augmentations_per_site=self.args.num_augmentations,
                mode="heuristic",
            )
        if self.args.random_placements:
            rotation_mode = (
                "random"
                if self.args.full_random_rotations
                else "random_site_heuristic_placement"
            )
            self.rand_adslabs = AdsorbateSlabConfig(
                self.slab,
                self.adsorbate,
                self.args.random_sites,
                self.args.num_augmentations,
                mode=rotation_mode,
            )

        # write files
        if not args.skip_surface_inputs:
            write_surface(self.args, self.slab, self.bulk_index, self.surface_index)
        if self.heur_adslabs:
            self._write_adslabs(self.heur_adslabs, "heur")
        if self.rand_adslabs:
            prefix = "rand" if rotation_mode == "random" else "randsh"
            self._write_adslabs(self.rand_adslabs, prefix)

        end = time.time()
        self.logger.info(
            f"Completed adsorbate {self.adsorbate_index}, bulk {self.bulk_index}, surface {self.surface_index} ({round(end - start, 2)}s)"
        )

    def _write_adslabs(self, adslab_obj, mode_str):
        """
        Write one set of adslabs (called separately for random and heurstic placements)
        """
        for adslab_ind, adslab_atoms in enumerate(adslab_obj.atoms_list):
            adslab_dir = os.path.join(
                self.args.output_dir,
                f"bulk{self.bulk_index}",
                f"surface{self.surface_index}",
                f"ads{self.adsorbate.adsorbate_id_from_db}",
                f"{mode_str}{adslab_ind}",
            )

            # vasp files
            write_vasp_input_files(adslab_atoms, adslab_dir)

            # write dict for metadata
            metadata_path = os.path.join(adslab_dir, "metadata.pkl")
            metadata_dict = adslab_obj.get_metadata_dict(adslab_ind)
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata_dict, f)

            if self.args.no_vasp:
                # A bit hacky but ASE defaults to writing everything out.
                for unused_file in ["KPOINTS", "INCAR", "POTCAR"]:
                    unused_file_path = os.path.join(adslab_dir, unused_file)
                    if os.path.isfile(unused_file_path):
                        os.remove(unused_file_path)


def write_surface(args, slab, bulk_index, surface_index):
    """
    Writes vasp inputs and metadata for a specified  slab
    """

    os.makedirs(os.path.join(args.output_dir, f"bulk{bulk_index}"), exist_ok=True)
    os.makedirs(
        os.path.join(
            args.output_dir,
            f"bulk{bulk_index}",
            f"surface{surface_index}",
        ),
        exist_ok=True,
    )

    # write vasp files
    slab_alone_dir = os.path.join(
        args.output_dir,
        f"bulk{bulk_index}",
        f"surface{surface_index}",
        "surface",
    )
    if not os.path.exists(os.path.join(slab_alone_dir, "POSCAR")):
        # Skip surface if already written;
        # this happens when we process multiple adsorbates per surface.
        write_vasp_input_files(slab.atoms, slab_alone_dir)

    # write metadata
    metadata_path = os.path.join(
        args.output_dir,
        f"bulk{bulk_index}",
        f"surface{surface_index}",
        "surface",
        "metadata.pkl",
    )
    if not os.path.exists(metadata_path):
        metadata_dict = slab.get_metadata_dict()
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata_dict, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample adsorbate and bulk surface(s)")

    # input databases
    parser.add_argument(
        "--bulk_db", type=str, required=True, help="Underlying db for bulks (.pkl)"
    )
    parser.add_argument(
        "--adsorbate_db",
        type=str,
        help="Underlying db for adsorbates (.pkl)",
    )

    # Slabs for each bulk - if provided, this will save computation,
    # otherwise the slabs will be generated and saved out
    parser.add_argument(
        "--precomputed_slabs_dir",
        type=str,
        default=None,
        help="Root directory of precomputed surfaces",
    )

    # material specifications, option A: provide one set of indices
    parser.add_argument(
        "--adsorbate_index", type=int, default=None, help="Adsorbate index (int)"
    )
    parser.add_argument(
        "--bulk_index",
        type=int,
        default=None,
        help="Bulk index (int)",
    )
    parser.add_argument(
        "--surface_index", type=int, default=None, help="Surface index (int)"
    )

    # material specifications, option B: provide one set of indices
    parser.add_argument(
        "--indices_file",
        type=str,
        default=None,
        help="File containing adsorbate_bulk_surface indices",
    )
    parser.add_argument(
        "--bulk_indices_file",
        type=str,
        default=None,
        help="If indices_file not provided, file containing bulk indices to precompute slabs",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=1,
        help="For multi-node processing, number of chunks to split inputs across.",
    )
    parser.add_argument(
        "--chunk_index",
        type=int,
        default=0,
        help="For multi-node processing, index of chunk to process.",
    )

    # output
    parser.add_argument("--output_dir", type=str, help="Root directory for outputs")

    # other options
    parser.add_argument(
        "--max_miller",
        type=int,
        default=2,
        help="Max miller indices to consider for generating surfaces",
    )
    parser.add_argument(
        "--random_placements",
        action="store_true",
        default=False,
        help="Generate random placements",
    )
    parser.add_argument(
        "--full_random_rotations",
        action="store_true",
        default=False,
        help="Random placements have full rotation around the sphere, as opposed to small wobbles around x&y",
    )
    parser.add_argument(
        "--heuristic_placements",
        action="store_true",
        default=False,
        help="Generate heuristic placements",
    )
    parser.add_argument(
        "--random_sites",
        type=int,
        default=None,
        help="Number of random placement per adsorbate/surface if args.random_placements is set to True",
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=1,
        help="Number of random augmentations (i.e. rotations) per site",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling/random sites generation",
    )
    parser.add_argument(
        "--no_vasp",
        action="store_true",
        default=False,
        help="Do not write out POTCAR/INCAR/KPOINTS for adslabs",
    )
    parser.add_argument(
        "--skip_surface_inputs",
        action="store_true",
        default=False,
        help="Skip writing DFT surface inputs",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Log detailed info"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of workers for multiprocessing when given a file of indices",
    )

    args = parser.parse_args()

    # check that all needed args are supplied
    if args.indices_file:
        if not (args.random_placements or args.heuristic_placements):
            parser.error("Must choose either or both of random or heuristic placements")
        if args.random_placements and (
            args.random_sites is None or args.random_sites <= 0
        ):
            parser.error("Must specify number of sites for random placements")
    elif args.bulk_indices_file:
        assert args.precomputed_slabs_dir is not None
    else:
        if (
            args.adsorbate_index is None
            or args.bulk_index is None
            or args.surface_index is None
        ):
            parser.error("Must provide a file or specify all material indices")

    return args


def precompute_slabs(bulk_ind):
    try:
        bulk = Bulk(bulk_id_from_db=int(bulk_ind), bulk_db_path=args.bulk_db)
        all_slabs = bulk.get_slabs(
            max_miller=args.max_miller,
            precomputed_slabs_dir=args.precomputed_slabs_dir,
        )
        if not args.no_vasp:
            for surf_idx, slab in enumerate(all_slabs):
                write_surface(args, slab, bulk_ind, surf_idx)
    except Exception:
        traceback.print_exc()


def run_placements(inputs):
    args, ads_ind, bulk_ind, surface_ind = inputs
    try:
        job = StructureGenerator(
            args,
            bulk_index=int(bulk_ind),
            surface_index=int(surface_ind),
            adsorbate_index=int(ads_ind),
        )
        job.run()
    except Exception:
        # Explicitly print errors or else the pool will fail silently
        traceback.print_exc()


if __name__ == "__main__":
    """
    This script creates adsorbate+surface placements and saves them out. An
    indices_file is required, which contains the adsorbate, bulk, and surface
    index desired for placement. Alternatively, if a bulk_indices_file is
    provided with bulk indices, slabs will be precomputed and saved to the
    provided directory.
    """
    args = parse_args()

    if args.indices_file:
        with open(args.indices_file, "r") as f:
            all_indices = f.read().splitlines()
        chunks = np.array_split(all_indices, args.chunks)
        inds_to_run = chunks[args.chunk_index]
        print(
            f"Running lines from {args.indices_file}, starting from {inds_to_run[0]} ending at {inds_to_run[-1]}"
        )

        pool_inputs = []
        for line in inds_to_run:
            ads_ind, bulk_ind, surface_ind = line.strip().split("_")
            pool_inputs.append((args, ads_ind, bulk_ind, surface_ind))

        pool = mp.Pool(args.workers)
        outputs = list(
            tqdm(pool.imap(run_placements, pool_inputs), total=len(pool_inputs))
        )
        pool.close()

        print("Placements successfully generated!")
    elif args.bulk_indices_file:
        with open(args.bulk_indices_file, "r") as f:
            all_indices = f.read().splitlines()

        chunks = np.array_split(all_indices, args.chunks)
        inds_to_run = chunks[args.chunk_index]

        pool = mp.Pool(args.workers)
        outputs = list(
            tqdm(pool.imap(precompute_slabs, inds_to_run), total=len(inds_to_run))
        )
        pool.close()

        print("Slabs successfully precomputed!")
    else:
        job = StructureGenerator(
            args,
            bulk_index=args.bulk_index,
            surface_index=args.surface_index,
            adsorbate_index=args.adsorbate_index,
        )
        job.run()

        print("Placements successfully generated!")
