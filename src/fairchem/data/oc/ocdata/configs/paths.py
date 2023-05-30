# Path to a database of bulks, organized as a list of dictionaries with
# each dict containing atoms objects, mpid, and other metadata.
BULK_PKL_PATH = "ocdata/databases/pkls/bulks.pkl"

# Path to a folder of pickle files, each containing a list of precomputed
# slabs. The filename of each pickle is <bulk_index.pkl> where `bulk_index`
# is the index of the corresponding bulk in BULK_PKL_PATH.
PRECOMPUTED_SLABS_DIR_PATH = (
    "/checkpoint/janlan/ocp/input_dbs/precomputed_surfaces_2021Sep20/"
)

# Path to a database of adsorbates, organized as a dictionary with a unique
# integer as key and corresponding adsorbate tuple as value.
ADSORBATES_PKL_PATH = "ocdata/databases/pkls/adsorbates.pkl"
