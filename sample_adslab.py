from ocdata.loader import Loader

with Loader("Imports"):
    import pickle
    from pathlib import Path

    import numpy as np
    from minydra import resolved_args

    from ocdata.adsorbates import Adsorbate
    from ocdata.bulk_obj import Bulk
    from ocdata.surfaces import Surface
    from ocdata.combined import Combined
    from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs


if __name__ == "__main__":
    with Loader("Full procedure", animate=False):
        # path to directory's root
        root = Path(__file__).resolve().parent
        # load default args then overwrite from command-line
        args = resolved_args(defaults=root / "configs" / "sample" / "defaults.yaml")
        # print parsed arguments
        if args.verbose > 0:
            args.pretty_print()

        # set seed
        np.random.seed(args.seed)
        with Loader(
            "Reading bulk_db_flat", animate=args.animate, ignore=args.no_loader
        ):
            # load flat bulk db
            with open(args.paths.bulk_db_flat, "rb") as f:
                bulk_db_list = pickle.load(f)

        print(
            "Surface sampling string:",
            "surface_idx / total_possible_surfaces_for_bulk",
        )

        for i in range(args.nruns):
            print("\n" + "-" * 30)
            print("-" * 30)
            with Loader(f"Actions to Data {i+1}/{args.nruns}", animate=False):

                with Loader(
                    "Make Adsorbate object", animate=args.animate, ignore=args.no_loader
                ):
                    # make Adsorbate object
                    # (adsorbate selection is done in the class if adsorbate_id is None)
                    adsorbate_obj = Adsorbate(
                        args.paths.adsorbate_db,
                        specified_index=args.actions.adsorbate_id,
                    )
                    print(
                        "# Selected adsorbate:",
                        adsorbate_obj.atoms.get_chemical_formula(),
                    )

                with Loader(
                    "Make Bulk object", animate=args.animate, ignore=args.no_loader
                ):

                    # select bulk_id if None
                    if args.actions.bulk_id is None:
                        bulk_id = np.random.choice(len(bulk_db_list))
                        print(f"args.actions.bulk_id is None, choosing {bulk_id}")
                    else:
                        bulk_id = args.actions.bulk_id

                    # make Bulk object
                    bulk = Bulk(
                        bulk_db_list,
                        bulk_index=bulk_id,
                        precomputed_structures=args.paths.precomputed_structures
                        if args.use_precomputed_surfaces
                        else None,
                    )
                    print(
                        "# Selected bulk:",
                        bulk.bulk_atoms.get_chemical_formula(),
                        f"({bulk.mpid})",
                    )

                with Loader(
                    "bulk.get_possible_surfaces()",
                    animate=args.animate,
                    ignore=args.no_loader,
                ):
                    possible_surfaces = bulk.get_possible_surfaces()

                with Loader(
                    "Make Surface object", animate=args.animate, ignore=args.no_loader
                ):
                    # select surface_id if it is None
                    if args.actions.surface_id is None:
                        surface_id = np.random.choice(len(possible_surfaces))
                        print(f"args.actions.surface_id is None, choosing {surface_id}")
                    else:
                        assert args.actions.surface_id < len(possible_surfaces)
                        surface_id = args.actions.surface_id

                    # make Surface object
                    surface_obj = Surface(
                        bulk,
                        possible_surfaces[surface_id],
                        surface_id,
                        len(possible_surfaces),
                    )
                    print(
                        "# Selected surface:",
                        surface_obj.surface_atoms.get_chemical_formula(),
                        f"({surface_obj.surface_sampling_str})",
                    )

                with Loader(
                    "Make Combined object", animate=args.animate, ignore=args.no_loader
                ):
                    # combine adsorbate + bulk
                    try:
                        adslab = Combined(
                            adsorbate_obj,
                            surface_obj,
                            enumerate_all_configs=False,
                            no_loader=args.no_loader,
                        )
                    except Exception as e:
                        print(str(e))
                        print("ABORTING")
                        continue
                    atoms_object = adslab.constrained_adsorbed_surfaces[0]

                with Loader(
                    "Make torch_geometric data",
                    animate=args.animate,
                    ignore=args.no_loader,
                ):
                    converter = AtomsToGraphs(
                        r_energy=False,
                        r_forces=False,
                        r_distances=True,
                        r_edges=True,
                        r_fixed=True,
                    )
                    # Convert ase.Atoms into torch_geometric.Data
                    data = converter.convert(atoms_object)
