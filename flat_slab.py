import catkit

from ocdata.constants import COVALENT_MATERIALS_MPIDS, MAX_MILLER
from ocdata.loader import Loader

with Loader("Imports"):
    import pickle
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    from minydra import resolved_args
    from pymatgen.core.surface import (
        SlabGenerator,
        get_symmetrically_distinct_miller_indices,
    )

    from ocdata.adsorbates import Adsorbate
    from ocdata.bulk_obj import Bulk
    from ocdata.combined import Combined
    from ocdata.surfaces import Surface
    from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs


# ----------------------------
# -----  UTILS (ignore)  -----
# ----------------------------


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


def print_out_times(out_times, fpath=None, prec=3):
    """
    Prints a summary of the out_time dictionnary

    Args:
        out_times (dict[list]): dictionnary of times
        fpath (Union[str, pathlib.Path], optional): path to write the
            string summary to. Defaults to None (= no writing)
        prec (int, optional): print decimals. Defaults to 3.

    Returns:
        str: stringsummary
    """
    max_k_len = max([len(k) for k in out_times])
    strs = [f"{'Operation':{max_k_len}} -> Time (s)"]

    all_keys = sorted(out_times.keys())
    single_keys = []
    if not all([len(k) == 1 for k in out_times]):
        single_keys = [k for k, v in out_times.items() if len(v) == 1]
        all_keys = single_keys + [k for k in out_times if k not in set(single_keys)]

    single_key_sep = None

    for i, k in enumerate(all_keys):
        if single_keys and k not in single_keys and single_key_sep is None:
            single_key_sep = i + 1
        times = out_times[k]
        s = f"{k:{max_k_len}} -> "
        if len(times) > 1:
            q1, med, q3 = np.percentile(times, [25, 50, 75])
            mean, std = np.mean(times), np.std(times)
            s += f"[{q1:.{prec}f} | {med:.{prec}f} | {q3:.{prec}f}]"
            s += f" ~ {mean:.{prec}f} +/- {std:.{prec}f}"
        else:
            s += f"{times[0]:.{prec}f}"
        strs.append(s)

    max_s_len = max(len(s) for s in strs)
    border = "-" * max_s_len
    strs.append(border)

    if single_key_sep is not None:
        strs = (
            strs[:single_key_sep]
            + [
                border,
                f"{'Operation':{max_k_len}} -> [q1 | med | q3] ~ mean +/- std",
                border,
            ]
            + strs[single_key_sep:]
        )

    out_str = "\n".join([border] + strs[:1] + [border] + strs[1:])

    if fpath is not None:
        with open(fpath, "w") as f:
            f.write(out_str)

    print(out_str)


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


# ------------------------------------
# -----  Action Space Functions  -----
# ------------------------------------


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
    root = Path(__file__).resolve().parent
    args = resolved_args(defaults=root / "configs" / "sample" / "defaults.yaml")
    if isinstance(
        args.actions.binding_site_index, str
    ) and args.actions.binding_site_index.lower() in {"null", "none"}:
        args.actions.binding_site_index = None

    out_times = defaultdict(list)
    seed = args.seed or 0

    with open(args.paths.bulk_db_flat, "rb") as f:
        bulk_db_list = pickle.load(f)

    ads_dict = get_ads_db(args)

    for i in range(args.nruns):
        np.random.seed(seed + i)
        run_loader = Loader(
            f"Actions to Data {i+1}/{args.nruns}", animate=False, out=out_times
        )
        run_loader.start()
        print_header(i, args.nruns)

        print("\n1. Adsorbate\n")
        adsorbate_atoms = select_adsorbate(ads_dict, args.actions.adsorbate_formula)
        adsorbate_obj = Adsorbate(adsorbate_atoms=adsorbate_atoms)  # <<<< IMPORTANT
        print("-> Selected adsorbate:", adsorbate_obj.atoms.get_chemical_formula())

        # ------------------
        # -----  Bulk  -----
        # ------------------

        print("\n2. Bulk\n")

        # select bulk_id if None
        if args.actions.bulk_id is None:
            bulk_id = np.random.choice(len(bulk_db_list))
            print(f"args.actions.bulk_id is None, choosing {bulk_id}")
        else:
            bulk_id = args.actions.bulk_id
        bulk = Bulk(  # <<<< IMPORTANT
            bulk_db_list,
            bulk_index=bulk_id,
            precomputed_structures=args.paths.precomputed_structures
            if args.use_precomputed_surfaces
            else None,
        )
        print(
            "-> Selected bulk:",
            bulk.bulk_atoms.get_chemical_formula(),
            f"({bulk.mpid})",
        )

        # possible_surfaces = bulk.get_possible_surfaces()
        # -> surfaces_info = self.enumerate_surfaces(
        #     miller_indices=None, sample_miller_indices=False, max_miller=MAX_MILLER
        # )
        bulk_struct = bulk.standardize_bulk(bulk.bulk_atoms)
        all_millers = get_symmetrically_distinct_miller_indices(bulk_struct, MAX_MILLER)
        np.random.shuffle(all_millers)

        all_slabs_info = []
        site_found = False  # at the begining, no binding site has been found

        for millers in all_millers:
            # 1. Select one set of miller indices

            if site_found:
                break

            # 2. Generate all possible slabs for this set of miller indices
            slab_gen = SlabGenerator(
                initial_structure=bulk_struct,
                miller_index=millers,
                min_slab_size=7.0,
                min_vacuum_size=20.0,
                lll_reduce=False,
                center_slab=True,
                primitive=True,
                max_normal_search=1,
            )
            slabs = slab_gen.get_slabs(
                tol=0.3, bonds=None, max_broken_bonds=0, symmetrize=False
            )
            # If the bottoms of the slabs are different than the tops, then we want
            # to consider them, too
            if len(slabs) != 0:
                flipped_slabs_info = [
                    (bulk.flip_struct(slab), millers, slab.shift, False)
                    for slab in slabs
                    if bulk.is_structure_invertible(slab) is False
                ]

                # Concatenate all the results together
                slabs_info = [(slab, millers, slab.shift, True) for slab in slabs]
                all_slabs_info.extend(slabs_info + flipped_slabs_info)

            if len(all_slabs_info) == 0:
                print("No surface found. Next Miller indices")
                continue
            else:
                print(
                    f"{len(all_slabs_info)} surfaces found for Miller indices {millers}"
                )

            possible_surfaces = all_slabs_info
            np.random.shuffle(possible_surfaces)

            # Try the current (miller indices, surface) combination and look for binding sites
            for surface in possible_surfaces:
                # 3. Select one surface

                surface_obj = Surface(
                    bulk,
                    surface,
                    0,  # dummy
                    0,  # dummy
                    no_loader=args.no_loader,
                )
                print(
                    "-> Selected surface:",
                    surface_obj.surface_atoms.get_chemical_formula(),
                )

                adslab = Combined(
                    adsorbate_obj,
                    surface_obj,
                    enumerate_all_configs=False,
                    no_loader=args.no_loader,
                    index=args.actions.binding_site_index,
                    early_init=True,
                )
                surface_gratoms = catkit.Gratoms(surface_obj.surface_atoms)
                surface_atom_indices = [
                    i
                    for i, atom in enumerate(surface_obj.surface_atoms)
                    if atom.tag == 1
                ]
                surface_gratoms.set_surface_atoms(surface_atom_indices)
                surface_gratoms.pbc = np.array([True, True, False])

                adsorbate_gratoms = adslab.convert_adsorbate_atoms_to_gratoms(
                    adsorbate_obj.atoms, adsorbate_obj.bond_indices
                )
                builder = catkit.gen.adsorption.Builder(surface_gratoms)

                # Try adding the adsorbate onto the first binding site of the
                # current surface

                for site_index in range(len(surface_atom_indices)):
                    adsorbed_surface = builder.add_adsorbate(
                        adsorbate_gratoms,
                        bonds=adsorbate_obj.bond_indices,
                        index=site_index,
                    )
                    is_reasonable = adslab.is_config_reasonable(adsorbed_surface)
                    if is_reasonable:
                        # use this binding site
                        site_found = True
                        break

                if site_found:
                    print("-> Site found")
                    break

                # all binding sites have been tried
                # go to next surface
                print("No more sites available. Next surface")
        run_loader.stop()
    print_out_times(out_times)
