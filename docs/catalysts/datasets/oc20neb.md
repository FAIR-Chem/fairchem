
# Open Catalyst 2020 Nudged Elastic Band (OC20NEB)

## Overview
This is a validation dataset which was used to assess model performance in [CatTSunami: Accelerating Transition State Energy Calculations with Pre-trained Graph Neural Networks](https://arxiv.org/abs/2405.02078). It is comprised of 932 NEB relaxation trajectories. There are three different types of reactions represented: desorptions, dissociations, and transfers. NEB calculations allow us to find transition states. The rate of reaction is determined by the transition state energy, so access to transition states is very important for catalysis research. For more information, check out the paper.

## File Structure and Contents
The tar file contains 3 subdirectories: dissociations, desorptions, and transfers. As the names imply, these directories contain the converged DFT trajectories for each of the reaction classes. Within these directories, the trajectories are named to identify the contents of the file. Here is an example and the anatomy of the name:

```desorption_id_83_2409_9_111-4_neb1.0.traj```

1. `desorption` indicates the reaction type (dissociation and transfer are the other possibilities)
2. `id` identifies that the material belongs to the validation in domain split (ood - out of domain is th e other possibility)
3. `83` is the task id. This does not provide relavent information
4. `2409` is the bulk index of the bulk used in the ocdata bulk pickle file
5. `9` is the reaction index. for each reaction type there is a reaction pickle file in the repository. In this case it is the 9th entry to that pickle file
6. `111-4` the first 3 numbers are the miller indices (i.e. the (1,1,1) surface), and the last number cooresponds to the shift value. In this case the 4th shift enumerated was the one used.
7. `neb1.0` the number here indicates the k value used. For the full dataset, 1.0 was used so this does not distiguish any of the trajectories from one another.


The content of these trajectory files is the repeating frame sets. Despite the initial and final frames not being optimized during the NEB, the initial and final frames are saved for every iteration in the trajectory. For the dataset, 10 frames were used - 8 which were optimized over the neb. So the length of the trajectory is the number of iterations (N) * 10. If you wanted to look at the frame set prior to optimization and the optimized frame set, you could get them like this:

```python
from ase.io import read

traj = read("desorption_id_83_2409_9_111-4_neb1.0.traj", ":")
unrelaxed_frames = traj[0:10]
relaxed_frames = traj[-10:]
```

## Download 
|Splits |Size of compressed version (in bytes)  |Size of uncompressed version (in bytes)    | MD5 checksum (download link)   |
|---    |---    |---    |---    |
|ASE Trajectories   |1.5G  |6.3G   | [52af34a93758c82fae951e52af445089](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20neb/oc20neb_dft_trajectories_04_23_24.tar.gz)   |



## Use
One more note: We have not prepared an lmdb for this dataset. This is because it is NEB calculations are not supported directly in ocp. You must use the ase native OCP class along with ase infrastructure to run NEB calculations. Here is an example of a use:

```python
from ase.io import read
from ase.optimize import BFGS
from fairchem.applications.cattsunami.core import OCPNEB

traj = read("desorption_id_83_2409_9_111-4_neb1.0.traj", ":")
neb_frames = traj[0:10]
neb = OCPNEB(
    neb_frames,
    checkpoint_path=YOUR_CHECKPOINT_PATH,
    k=k,
    batch_size=8,
)
optimizer = BFGS(
    neb,
    trajectory=f"test_neb.traj",
)
conv = optimizer.run(fmax=0.45, steps=200)
if conv:
    neb.climb = True
    conv = optimizer.run(fmax=0.05, steps=300)
```
