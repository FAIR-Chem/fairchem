"""
Script to generate gifs from traj

Requirements:

sudo apt update
sudo apt install povray
sudo apt install ffmpeg
pip install ase==3.20

"""
import os
import copy
import numpy as np
import multiprocessing as mp

import ase.io
from ase.data import covalent_radii
from ase.io.pov import get_bondpairs

def pov_from_atoms(mp_args):

    atoms, idx, out_path = mp_args
    #how many extra repeats to generate on either side to look infinite
    extra_cells=2
    # try and guess which atoms are adsorbates since the tags aren't correct after running in vasp
    # ideally this would be fixed by getting the right adsorbate atoms from the initial configurations
    atoms_organic = np.array([atom.symbol in set(['C','H','O','N']) for atom in atoms])
    # get the bare surface (note: this will not behave correctly for nitrides/hydrides/carbides/etc)
    atoms_surface = atoms[~atoms_organic].copy()
    #replicate the bare surface
    atoms_surface = atoms_surface.repeat((extra_cells*2+1,extra_cells*2+1,1))
    # make an image of the adsorbate in the center of the slab
    atoms_adsorbate = atoms[atoms_organic]
    atoms_adsorbate.positions += extra_cells*(atoms.cell[0,:] + atoms.cell[1,:])
    #add the adsorbate to the replicated surface, then center the positions on the adsorbate
    num_surface_atoms = len(atoms_surface)
    atoms_surface+=atoms_adsorbate
    atoms_surface.positions-=atoms_adsorbate.positions.mean(axis=0)
    #only include bonds for the adsorbate atoms
    bondpairs = get_bondpairs(atoms_surface)
    bondpairs = [bond for bond in bondpairs if bond[0]>=num_surface_atoms and bond[1]>=num_surface_atoms]
    # write the image with povray
    bbox=(-6.4,-4,6.4,4) #clip to a small region around the adsorbate
    atoms_surface.write(f'{out_path}/snapshot_%04i.pov'%idx,
                        run_povray=True,
                        bbox=bbox,
                        celllinewidth=0,
                        rotation='-40x',
                        canvas_height=300,
                        textures=['intermediate']*len(atoms_surface),
                        bondatoms=bondpairs,
                        radii=covalent_radii[atoms_surface.numbers])
    print(f"image {idx} completed!")

def parallelize_generation(traj_path, out_path=".", n_procs=8):

    # make the covalent radii for O/C/N a little smaller to make bonds visible
    covalent_radii[6]=covalent_radii[6]*0.7
    covalent_radii[7]=covalent_radii[7]*0.7
    covalent_radii[8]=covalent_radii[8]*0.7

    # name of the folder containing images and gif
    file_name = os.path.basename(traj_path).split(".")[0]
    out_path = os.path.join(out_path, file_name)
    os.makedirs(out_path, exist_ok=True)

    atoms_list = ase.io.read(traj_path, ":")

    # parallelizing image generation
    mp_args_list = [(atoms, idx, out_path) for idx, atoms in enumerate(atoms_list)]
    pool = mp.Pool(processes=n_procs)
    pool.map(pov_from_atoms, mp_args_list)

    # creating gif
    os.system(f"ffmpeg -pattern_type glob -i '{out_path}/*.png' {out_path}/{file_name}.gif")

if __name__ == "__main__":

    traj_path = "random2574092.traj"
    parallelize_generation(traj_path)
