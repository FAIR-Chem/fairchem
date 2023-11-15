from pymatgen.core.structure import Structure
from pymatgen.core.surface import SlabGenerator

# Import the LiFePO4 structure
LiFePO4 = Structure.from_file("ocdata/LiFePO4.cif")
print(LiFePO4)

# Let's add some oxidation states to LiFePO4
LiFePO4.add_oxidation_state_by_element({"Fe": 2, "Li": 1, "P": 5, "O": -2})

# Generate the slab = cut through the surface using Miller indices
slabgen = SlabGenerator(
    LiFePO4,
    miller_index=(0, 0, 1),
    min_slab_size=10,
    min_vacuum_size=10,
    center_slab=True,
)
# Get all the unique terminations along the normal to the Miller plane we are interested in
slabs = slabgen.get_slabs()
print(f"There are {len(slabs)} slabs")
print(slabs[0])
