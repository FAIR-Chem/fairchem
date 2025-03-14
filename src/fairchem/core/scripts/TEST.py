import os
import download_large_files
from fairchem.data.oc.databases.pkls import BULK_PKL_PATH

# Test calling the function and installing the bulk.pkl file; analogous
# to init() in fairchem.data.oc.core.bulk 
 
if not os.path.exists(BULK_PKL_PATH):
    download_large_files.download_file_group("oc")
else:
    print(f"Path to bulk .pkl file already exists, under: {BULK_PKL_PATH}")

