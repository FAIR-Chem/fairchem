
import os
import pickle
import requests
from tqdm import tqdm

API_ROOT = "https://www.materialsproject.org/rest/v2/materials/"
API_KEY = "XRpCxuAEdfTHIhtw"
API_CIF_SUFFIX = "/vasp/cif"

DATA_FILE = "/srv/share2/abhshkdz/data/electrocatalyst-design/2019_09_19/docs.pkl"
DATA_OUTPUT_DIR = "/srv/share2/abhshkdz/data/electrocatalyst-design/2019_09_19/energy-regression/"


if __name__ == "__main__":
    docs = pickle.load(open(DATA_FILE, "rb"))
    print("Total docs", len(docs))

    # caching already made requests
    mpid2cif = {}

    for i in tqdm(docs):
        mongo_id = str(i["mongo_id"])
        mp_id = i["mpid"]

        if mp_id not in mpid2cif:
            # Query the API to get CIF.
            r = requests.get(
                url=API_ROOT + mp_id + API_CIF_SUFFIX,
                params={"API_KEY": API_KEY})

            data = r.json()

            if len(data["response"]) == 0:
                continue

            try:
                assert len(data["response"]) == 1
            except AssertionError as error:
                import pdb; pdb.set_trace()

            cif = data["response"][0]["cif"]
            mpid2cif[mp_id] = cif
        else:
            cif = mpid2cif[mp_id]

        # Write CIF.
        with open(os.path.join(DATA_OUTPUT_DIR, str(mongo_id) + ".cif"), "w") as f:
            f.write(cif)

        # Write target.
        with open(os.path.join(DATA_OUTPUT_DIR, "id_prop.csv"), "a+") as f:
            f.write(mongo_id + "," + str(i["results"]["energy"]) + "\n")
