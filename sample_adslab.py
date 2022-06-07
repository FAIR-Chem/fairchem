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


def print_header(i, nruns):
    """
    Prints
    -------------------
    ----   Run i   ----
    -------------------
    """
    box_char = "#"
    border_width = 4
    border = box_char * border_width
    box_width = 40

    runs_len = len(str(nruns))
    title_str = f"Run {str(i + 1).zfill(runs_len)}/{nruns}"

    n_space = box_width - 2 * len(border) - len(title_str)
    n_left = n_space // 2
    n_right = n_space // 2 + (n_space % 2)

    print("\n" + box_char * box_width)
    print(border + " " * n_left + title_str + " " * n_right + border)
    print(box_char * box_width)


def get_ads_db(args):
    """
    Util to load the adsorbates pre-computed dict from the args

    Args:
        args (Union[dict, minydra.MinyDict]): Command-line args

    Returns:
        dict: adsorbates dictionnary
    """
    with open(args.paths.adsorbate_db, "rb") as f:
        return pickle.load(f)


def select_adsorbate(ads_dict, smiles):
    """
    Function to parameterize the choice of an adsorbate.
    Curent parameterization relies on its chemical formula.

    Args:
        db_path (Union[str, pathlib.Path]): path to the pickle file holding adsorbates
        smiles (str): The smiles string description for the adsorbate

    Returns:
        Optional[ase.Atom]: The selected adsorbate. None if the formula does not exist
    """

    if smiles is None:
        smiles = np.random.choice([a[1] for a in ads_dict.values()])
        print(
            "No adsorbate smiles has been provided. Selecting {} at random.".format(
                smiles
            )
        )

    adsorbates = [(str(k), *a) for k, a in ads_dict.items() if a[1] == smiles]

    if len(adsorbates) == 0:
        raise ValueError(f"No adsorbate exists with smiles {smiles}")
    if len(adsorbates) > 1:
        raise ValueError(
            f"More than 1 adsorbate exists with smiles {smiles}:\n"
            + ", ".join([a[2] for a in adsorbates])
        )

    return adsorbates[0]


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

        with Loader(
            "Reading adsorbates dict", animate=args.animate, ignore=args.no_loader
        ):
            ads_dict = get_ads_db(args)

        print(
            "Surface sampling string:",
            "surface_idx / total_possible_surfaces_for_bulk",
        )

        for i in range(args.nruns):

            print_header(i, args.nruns)

            with Loader(f"Actions to Data {i+1}/{args.nruns}", animate=False):

                print("\n1. Adsorbate\n")

                with Loader(
                    "Make Adsorbate object", animate=args.animate, ignore=args.no_loader
                ):
                    adsorbate_atoms = select_adsorbate(
                        ads_dict, args.actions.adsorbate_formula
                    )
                    # make Adsorbate object
                    # (adsorbate selection is done in the class if adsorbate_id is None)
                    adsorbate_obj = Adsorbate(  # <<<< IMPORTANT
                        adsorbate_atoms=adsorbate_atoms
                    )
                    print(
                        "# Selected adsorbate:",
                        adsorbate_obj.atoms.get_chemical_formula(),
                    )

                print("\n2. Bulk\n")

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
                    bulk = Bulk(  # <<<< IMPORTANT
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

                print("\n3. Surface\n")

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
                    surface_obj = Surface(  # <<<< IMPORTANT
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

                print("\n4. Combined\n")

                with Loader(
                    "Make Combined object", animate=args.animate, ignore=args.no_loader
                ):
                    # combine adsorbate + bulk
                    try:
                        adslab = Combined(  # <<<< IMPORTANT
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

                print("\n5. Data\n")

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
