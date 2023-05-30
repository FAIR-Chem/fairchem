import glob
import multiprocessing
import os
import shutil
import sys
import zipfile
from datetime import datetime

import ase.io


def extract_file(zipname, file_to_unzip, extract_to):
    with zipfile.ZipFile(zipname, "r") as traj_zip:
        traj_zip.extract(file_to_unzip, extract_to)


def process_func(indices, dirlist, ans):
    for i in indices:
        dir_name = dirlist[i]

        extract_file(dir_name + "/relaxation_outputs.zip", "vasprun.xml", dir_name)
        trajname = dir_name + "/vasprun.xml"
        full_traj = ase.io.read(filename=trajname, index=":", format="vasp-xml")
        val = full_traj[-1].get_potential_energy(apply_constraint=False)
        print(dir_name, val)
        ans.append((dir_name, val))


if __name__ == "__main__":
    input_folder = "temp_download/"
    output = "ans/"
    id = sys.argv[1]

    dirlist = glob.glob(input_folder + "*/")

    manager = multiprocessing.Manager()
    ans = manager.list()

    k = multiprocessing.cpu_count()

    indices = [i for i in range(len(dirlist))]
    tasks = [indices[i::k] for i in range(k)]
    procs = []

    # instantiating processes
    for t in tasks:
        proc = multiprocessing.Process(target=process_func, args=(t, dirlist, ans))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

    # write to file
    with open(output + str(id) + ".txt", "w") as f:
        for i in range(len(ans)):
            name = ans[i][0].split("/")[-2]
            val = str(ans[i][1])
            f.write(name + " " + val + "\n")
