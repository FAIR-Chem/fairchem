# README_is2res

(
    This file was obtained after running: 
    ```
    python python scripts/download_data.py --task is2re --input-path /network/datasets/oc20 --data-path /home/mila/s/schmidtv/scratch/ocp-scratch --num-workers 4
    ```
)

This readme describes the file details for the IS2RE/IS2RS tasks and all its subsequent data splits.
 
This folder contains files organized as follows:

`data/is2re/10k/train/data.lmdb`
`data/is2re/100k/train/data.lmdb`
`data/is2re/all/train/data.lmdb`
`data/is2re/all/val_id/data.lmdb`
`data/is2re/all/val_ood_ads/data.lmdb`
`data/is2re/all/val_ood_cat/data.lmdb`
`data/is2re/all/val_ood_both/data.lmdb`
`data/is2re/all/test_id/data.lmdb`
`data/is2re/all/test_ood_ads/data.lmdb`
`data/is2re/all/test_ood_cat/data.lmdb`
`data/is2re/all/test_ood_both/data.lmdb`

There is additional  `.lmdb-lock` file present alongside each `.lmdb` file.

`data/is2re/N/M/data.lmdb` is an LMDB file containing N [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/data.html#Data) objects from adsorbate+catalyst systems in the corresponding M split. Each LMDB shall contain the following number of Data objects:

`data/is2re/10k/train/data.lmdb:            10,000`
`data/is2re/100k/train/data.lmdb:          100,000`
`data/is2re/all/train/data.lmdb:           460,328`
`data/is2re/all/val_id/data.lmdb:           24,943`
`data/is2re/all/val_ood_ads/data.lmdb:      24,961`
`data/is2re/all/val_ood_cat/data.lmdb:      24,963`
`data/is2re/all/val_ood_both/data.lmdb:     24,987`
`data/is2re/all/test_id/data.lmdb:          24,948`
`data/is2re/all/test_ood_ads/data.lmdb:     24,930`
`data/is2re/all/test_ood_cat/data.lmdb:     24,965`
`data/is2re/all/test_ood_both/data.lmdb:    24,985`

Each Data object includes the following information for each corresponding system (assuming K atoms):

* `sid` - [1] System ID corresponding to each structure
* `edge_index` - [2 x  J] Graph connectivity with index 0 corresponding to neighboring atoms and index 1 corresponding to center atoms. J corresponds to the total edges as determined by a nearest neighbor search.
* `atomic_numbers` - [K x 1] Atomic numbers of all atoms in the system
* `pos` - [K x 3] Initial structure positional information of all atoms in the system (x, y, z cartesian coordinates)
* `natoms` - [1] Total number atoms in the system
* `cell` -  [3  x 3] System unit cell (necessary for periodic boundary condition (PBC) calculations)
* `cell_offsets` - [J x 3] offset matrix where each index corresponds to the unit cell offset necessary to find the corresponding neighbor in  `edge_index`. For example,  `cell_offsets[0, :] = [0,1,0]` corresponds to `edge_index[:, 0]= [1,0]` representing node 1 as node 0’s neighbor located one unit cell over in the +y direction.
* `tags` - [K x 1] Atomic tag information: 0 - Fixed, sub-surface atoms, 1 - Free, surface atoms 2 - Free, adsorbate atoms

Train/Val LMDBs additionally contain the following attributes:

* `y_init` - [1] Initial structure energy of the system
* `y_relaxed` - [1] Relaxed structure energy of the system
* `pos_relaxed` - [K x 3] Relaxed structure positional information of all atoms in the system (x, y, z cartesian coordinates)


This LMDB file requires no additional processing and is ready to be used directly with the repository’s Datasets and DataLoaders. Move  `data/` directory to your project root directory.

