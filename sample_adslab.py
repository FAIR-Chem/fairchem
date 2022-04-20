from ocdata.loader import Loader

with Loader("Imports"):
    import numpy as np
    from ocdata.adsorbates import Adsorbate
    from ocdata.bulk_obj import Bulk
    from ocdata.surfaces import Surface
    from ocdata.combined import Combined
    import pickle
    from minydra import resolved_args
    from pathlib import Path
    from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs


if __name__ == "__main__":

    # path to directory's root
    root = Path(__file__).resolve().parent
    # load default args then overwrite from command-line
    args = resolved_args(defaults=root / "configs" / "sample" / "defaults.yaml")
    # print parsed arguments
    args.pretty_print()

    # set seed
    np.random.seed(args.seed)

    # load flat bulk db
    with open(args.paths.bulk_db_flat, "rb") as f:
        bulk_db_list = pickle.load(f)

    # select bulk_id if None
    if args.bulk_id is None:
        args.bulk_id = np.random.choice(len(bulk_db_list))
        print(f"args.bulk_id is None, choosing {args.bulk_id}")

    # make Bulk object
    bulk = Bulk(
        bulk_db_list,
        bulk_index=args.bulk_id,
        precomputed_structures=args.paths.precomputed_structures
        if args.use_precomputed_surfaces
        else None,
    )

    # select surface_id if it is None
    if args.surface_id is None:
        with Loader("Get possible surfaces"):
            possible_surfaces = bulk.get_possible_surfaces()
            args.surface_id = np.random.choice(len(possible_surfaces))
        print(f"args.surface_id is None, choosing {args.surface_id}")
    else:
        assert args.surface_id < len()

    # make Surface object
    surface = Surface(
        bulk,
        possible_surfaces[args.surface_id],
        args.surface_id,
        len(possible_surfaces),
    )

    # make Adsorbate object
    # (adsorbate selection is done in the class if adsorbate_id is None)
    adsorbate_obj = Adsorbate(
        args.paths.adsorbate_db,
        specified_index=args.adsorbate_id,
    )
    with Loader("Combine adslab"):
        # combine adsorbate + bulk
        adslab = Combined(adsorbate_obj, surface, enumerate_all_configs=False)
        atoms_object = adslab.constrained_adsorbed_surfaces[0]

    with Loader("Make torch_geometric data"):
        converter = AtomsToGraphs(
            r_energy=False,
            r_forces=False,
            r_distances=True,
            r_edges=True,
            r_fixed=True,
        )
        # Convert ase.Atoms into torch_geometric.Data
        data = converter.convert(atoms_object)
