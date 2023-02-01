"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

"""


OC20_elements = [
    "Pb",
    "Cs",
    "Nb",
    "In",
    "Ta",
    "Sc",
    "Ni",
    "Ru",
    "Fe",
    "Au",
    "Sb",
    "As",
    "Cu",
    "Ir",
    "Al",
    "N",
    "Rb",
    "Hf",
    "V",
    "Ca",
    "P",
    "Sr",
    "Na",
    "S",
    "Si",
    "Zn",
    "C",
    "Cd",
    "Ti",
    "Co",
    "Ge",
    "Mo",
    "H",
    "Pt",
    "Os",
    "O",
    "K",
    "Tl",
    "Ag",
    "Y",
    "Bi",
    "Rh",
    "Cl",
    "Ga",
    "W",
    "Cr",
    "Tc",
    "Sn",
    "B",
    "Pd",
    "Mn",
    "Hg",
    "Re",
    "Te",
    "Zr",
    "Se",
]
OC22_elements = [
    "Pb",
    "Nb",
    "Cs",
    "In",
    "Ta",
    "Sc",
    "Ni",
    "Fe",
    "Ru",
    "Au",
    "Ba",
    "Sb",
    "Cu",
    "Ir",
    "As",
    "Al",
    "Be",
    "N",
    "Rb",
    "V",
    "Hf",
    "Ca",
    "Sr",
    "Ce",
    "Na",
    "Cd",
    "Si",
    "Zn",
    "Li",
    "C",
    "Ti",
    "Ge",
    "Mo",
    "Co",
    "Pt",
    "Os",
    "H",
    "O",
    "K",
    "Tl",
    "Ag",
    "Y",
    "Mg",
    "Bi",
    "Rh",
    "Ga",
    "W",
    "Cr",
    "Sn",
    "Pd",
    "Hg",
    "Lu",
    "Mn",
    "Re",
    "Te",
    "Zr",
    "Se",
]


def interpolate_embeddings(
    checkpoint_atom_embeddings,
    fitted_datasets=["OC20", "OC22"],
    additional_fitted_elements=None,
    smoothing=0,
):
    # checkpoint_atom_embedding is tensor of shape (N_elements, embedding_size)
    # fitted_datasets is a list of strings that represent stanard OCP datasets that the checkpoint might
    #       have been trained on. It is used to guess which elements should be trusted
    # additional_fitted_elements is an an additional list of elements that were fitted, perhaps in case the dataset
    #       that the dataset that this was trained on was not OC20 or OC22
    # smoothing is an optional parameter to pass to the interpolator. 0 preserves perfect lookup for the trusted
    #       embeddings, and a value>0 will lead to smoothing of the embeddings across the periodic table

    from pymatgen.core.periodic_table import Element
    from scipy.interpolate import RBFInterpolator

    # Get all of the elements we should trust in checkpoint_atom_embeddings
    fitted_elements = set()
    if "OC20" in fitted_datasets:
        fitted_elements = fitted_elements.union(OC20_elements)
    if "OC22" in fitted_datasets:
        fitted_elements = fitted_elements.union(OC22_elements)
    if additional_fitted_elements is not None:
        fitted_elements = fitted_elements.union(additional_fitted_elements)
    fitted_elements = [Element(i) for i in fitted_elements]

    def determine_pseudo_row_group(el):
        # From the from_row_and_group fn in pmg Element class
        if 57 <= el.Z <= 71:
            el_pseudorow = 8
            el_pseudogroup = (el.Z - 54) % 32
        elif 89 <= el.Z <= 103:
            el_pseudorow = 9
            el_pseudogroup = (el.Z - 54) % 32
        else:
            el_pseudorow = el.row
            el_pseudogroup = el.group
        return el_pseudorow, el_pseudogroup

    all_elements = [
        determine_pseudo_row_group(Element.from_Z(i))
        for i in range(1, checkpoint_atom_embeddings.shape[0] + 1)
    ]
    interpolated_embeddings = RBFInterpolator(
        [determine_pseudo_row_group(el) for el in fitted_elements],
        checkpoint_atom_embeddings[
            [el.Z - 1 for el in fitted_elements], :
        ].cpu(),
        smoothing=smoothing,
    )(all_elements)

    return interpolated_embeddings
