import os
import random
import shutil
import sys
import zipfile
from datetime import datetime

import ase.io


def main():
    """
    Rattles every image along a relaxation pathway at 5 different variances.
    Rattled images are then put in their own directory along with the input
    files necessary to run VASP calculations.
    """
    random.seed(7361483614)
    root_dir = "."
    all_adsorbates = os.listdir(root_dir)
    all_adsorbates.remove("18")

    for adsorbate in all_adsorbates:
        try:
            int(adsorbate)
        except Exception:
            continue
        runs = os.listdir(os.path.join(root_dir, adsorbate))
        for run in runs:
            specific_names = os.listdir(os.path.join(root_dir, adsorbate, run))
            for fair_name in specific_names:
                current_dir = os.path.join(root_dir, adsorbate, run, fair_name)
                full_traj = ase.io.read(
                    current_dir + f"/{fair_name}_{adsorbate}_full.traj", ":"
                )
                for idx, image in enumerate(full_traj):
                    for idx_var, var in enumerate([0.005, 0.01, 0.05, 0.1, 0.5]):
                        rattled_image = image.copy()
                        rattled_image.rattle(stdev=var, seed=8472589)
                        to_dir = f"./perturbed_structures/{adsorbate}/{fair_name}/step_{idx}/{var}"
                        inputs = f"./input_files/{adsorbate}/{run}/"
                        shutil.copytree("./base_inputs/", to_dir)
                        shutil.copy(inputs + "KPOINTS", to_dir)
                        shutil.copy(inputs + "POTCAR", to_dir)
                        ase.io.write(
                            os.path.join(to_dir, "POSCAR"), rattled_image, format="vasp"
                        )
                        with open("./base_inputs/SUB_vasp.sh", "r") as file:
                            data = file.readlines()
                            data[3] = f"#SBATCH --job-name={run}_{idx}_{idx_var}\n"
                            data[4] = f"#SBATCH --output={run}_{idx}_{idx_var}.out\n"
                            data[6] = f"#SBATCH --error={run}_{idx}_{idx_var}.err\n"
                        with open(os.path.join(to_dir, "SUB_vasp.sh"), "w") as file:
                            file.writelines(data)
                            file.close()


if __name__ == "__main__":
    main()
