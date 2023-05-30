import os
import sys
import zipfile
from datetime import datetime

import ase.io


def extract_file(zipname, file_to_unzip, extract_to):
    with zipfile.ZipFile(zipname, "r") as traj_zip:
        traj_zip.extract(file_to_unzip, extract_to)


def main():
    """
    Given a directory containing adsorbate subdirectories, loops through all
    runs and merges intermediate checkpoints into a single, full trajectory.
    """
    # TODO: Impove efficiency for when dealing with larger systems
    root_dir = sys.argv[1]
    all_adsorbates = os.listdir(root_dir)

    for adsorbate in all_adsorbates:
        try:
            # Check if directory is an adsorbate
            int(adsorbate)
        except Exception:
            break
        runs = os.listdir(os.path.join(root_dir, adsorbate))
        for run in runs:
            specific_names = os.listdir(os.path.join(root_dir, adsorbate, run))
            # Redundant directory?
            for fair_name in specific_names:
                current_dir = os.path.join(root_dir, adsorbate, run, fair_name)
                checkpoint_dir = os.path.join(current_dir, "checkpoints")
                ordered_files = []
                if os.path.isdir(checkpoint_dir):
                    pass
                else:
                    print(current_dir)
                if os.path.isdir(checkpoint_dir):
                    # Sort checkpoint files
                    checkpoint_files = os.listdir(checkpoint_dir)
                    sorted_checkpoints = sorted(
                        checkpoint_files,
                        key=lambda x: datetime.strptime(
                            checkpoint_files[0][11:-4], "%Y-%m-%dT%H:%M:%S.%f"
                        ),
                    )
                    for idx, checkpoint in enumerate(sorted_checkpoints):
                        # Extract vasprun.xml file from each checkpoint file
                        cp_name = checkpoint_dir + f"/{checkpoint}"
                        extract_file(cp_name, "vasprun.xml", current_dir)
                        saved_name = current_dir + f"/checkpoint_{idx}"
                        os.rename(current_dir + "/vasprun.xml", saved_name)
                        ordered_files.append(saved_name)
                    # Extract vasprun.xml file from final checkpoint
                    final_name = current_dir + "/relaxation_outputs.zip"
                    extract_file(final_name, "vasprun.xml", current_dir)
                    saved_name = current_dir + f"/checkpoint_{idx+1}"
                    os.rename(current_dir + "/vasprun.xml", saved_name)
                    # Read xml files and construct full ase trajectory file
                    ordered_files.append(saved_name)
                    for idx, traj in enumerate(ordered_files):
                        if idx == 0:
                            full_traj = ase.io.read(
                                filename=traj, index=":", format="vasp-xml"
                            )
                        else:
                            full_traj += ase.io.read(
                                filename=traj, index="1:", format="vasp-xml"
                            )
                        if idx == len(ordered_files) - 1:
                            ase.io.write(
                                current_dir + f"/{fair_name}_{adsorbate}_full.traj",
                                full_traj,
                            )
                        os.remove(traj)
                else:
                    # No checkpoint run
                    # Read xml file and construct ase trajectory
                    final_name = current_dir + "/relaxation_outputs.zip"
                    extract_file(final_name, "vasprun.xml", current_dir)
                    full_traj = ase.io.read(
                        filename=current_dir + "/vasprun.xml",
                        index=":",
                        format="vasp-xml",
                    )
                    ase.io.write(
                        current_dir + f"/{fair_name}_{adsorbate}_full.traj", full_traj
                    )
                    os.remove(current_dir + "/vasprun.xml")


if __name__ == "__main__":
    main()
