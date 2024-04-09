---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Mass inference
------------------

The ASE calculator is not necessarily the most efficient way to run a lot of computations. It is better to do a "mass inference" using a command line utility. We illustrate how to do that here. 

In this paper we computed about 10K different gold structures:

Boes, J. R., Groenenboom, M. C., Keith, J. A., & Kitchin, J. R. (2016). Neural network and Reaxff comparison for Au properties. Int. J. Quantum Chem., 116(13), 979–987. http://dx.doi.org/10.1002/qua.25115

You can retrieve the dataset below. In this notebook we learn how to do "mass inference" without an ASE calculator. You do this by creating a config.yml file, and running the `main.py` command line utility.

```{code-cell} ipython3
! [ ! -f data.db ]  && wget https://figshare.com/ndownloader/files/11948267 -O data.db 
```

```{code-cell} ipython3
! ase db data.db
```

You have to choose a checkpoint to start with. The newer checkpoints may require too much memory for this environment. 

```{code-cell} ipython3
from ocpmodels.common.model_registry import MODEL_REGISTRY
print(MODEL_REGISTRY.keys())

```

```{code-cell} ipython3
from ocpmodels.common.model_registry import model_name_to_local_file

checkpoint_path = model_name_to_local_file('GemNet-dT OC22', local_cache='/tmp/ocp_checkpoints/')
checkpoint_path

```

We have to update our configuration yml file with the dataset. It is necessary to specify the train and test set for some reason. 

```{code-cell} ipython3

from ocpmodels.common.tutorial_utils import generate_yml_config
yml = generate_yml_config(checkpoint_path, 'config.yml',
                   delete=['cmd', 'logger', 'task', 'model_attributes',
                           'dataset', 'slurm'],
                   update={'amp': True,
                           'gpus': 1,
                           'task.dataset': 'ase_db',
                           'task.prediction_dtype': 'float32',
                        
                           # Train data
                           'dataset.train.src': 'data.db',
                           'dataset.train.a2g_args.r_energy': False,
                           'dataset.train.a2g_args.r_forces': False,
                           'dataset.train.select_args.selection': 'natoms>5,xc=PBE',
                            # Test data - prediction only so no regression
                           'dataset.test.src': 'data.db',
                           'dataset.test.a2g_args.r_energy': False,
                           'dataset.test.a2g_args.r_forces': False,
                           'dataset.test.select_args.selection': 'natoms>5,xc=PBE',
                          })

yml
```

It is a good idea to redirect the output to a file. If the output gets too large here, the notebook may fail to save. Normally I would use a redirect like `2&>1`, but this does not work with the main.py method. An alternative here is to open a terminal and run it there.

```{code-cell} ipython3
%%capture inference
import time
t0 = time.time()
! python {ocp_main()} --mode predict --config-yml {yml} --checkpoint {checkpoint_path} --amp
print(f'Elapsed time = {time.time() - t0:1.1f} seconds')
```

```{code-cell} ipython3
! grep "Total time taken:" 'mass-inference.txt'
```

```{code-cell} ipython3
with open('mass-inference.txt', 'wb') as f:
    f.write(inference.stdout.encode('utf-8')) 
```

The mass inference approach takes 1-2 minutes to run. See the output [here](./mass-inference.txt).

```{code-cell} ipython3
results = ! grep "  results_dir:" mass-inference.txt
d = results[0].split(':')[-1].strip()
```

```{code-cell} ipython3
import numpy as np
results = np.load(f'{d}/s2ef_predictions.npz', allow_pickle=True)
results.files
```

It is not obvious, but the data from mass inference is not in the same order. We have to get an id from the mass inference, and then "resort" the results so they are in the same order.

```{code-cell} ipython3
inds = np.array([int(r.split('_')[0]) for r in results['ids']])
sind = np.argsort(inds)
inds[sind]
```

To compare this with the results, we need to get the energy data from the ase db.

```{code-cell} ipython3
from ase.db import connect
db = connect('data.db')

energies = np.array([row.energy for row in db.select('natoms>5,xc=PBE')])
natoms = np.array([row.natoms for row in db.select('natoms>5,xc=PBE')])
```

Now, we can see the predictions. The are only ok here; that is not surprising, the data set has lots of Au configurations that have never been seen by this model. Fine-tuning would certainly help improve this.

```{code-cell} ipython3
import matplotlib.pyplot as plt

plt.plot(energies / natoms, results['energy'][sind] / natoms, 'b.')
plt.xlabel('DFT')
plt.ylabel('OCP');
```

# The ASE calculator way

We include this here just to show that:

1. We get the same results
2. That this is much slower.

```{code-cell} ipython3
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
calc = OCPCalculator(checkpoint=os.path.expanduser(checkpoint_path), cpu=False)
```

```{code-cell} ipython3
import time
from tqdm import tqdm
t0 = time.time()
OCP, DFT = [], []
for row in tqdm(db.select('natoms>5,xc=PBE')):
    atoms = row.toatoms()
    atoms.set_calculator(calc)
    DFT += [row.energy / len(atoms)]
    OCP += [atoms.get_potential_energy() / len(atoms)]
print(f'Elapsed time {time.time() - t0:1.1} seconds')
```

This takes at least twice as long as the mass-inference approach above. It is conceptually simpler though, and does not require resorting.

```{code-cell} ipython3
plt.plot(DFT, OCP, 'b.')
plt.xlabel('DFT (eV/atom)')
plt.ylabel('OCP (eV/atom)');
```

# Comparing ASE calculator and main.py

The results should be the same. 

It is worth noting the default precision of predictions is float16 with main.py, but with the ASE calculator the default precision is float32. Supposedly you can specify `--task.prediction_dtype=float32` at the command line to or specify it in the config.yml like we do above, but as of the tutorial this does not resolve the issue.

As noted above (see also [Issue 542](https://github.com/Open-Catalyst-Project/ocp/issues/542)), the ASE calculator and main.py use different precisions by default, which can lead to small differences. 

```{code-cell} ipython3
np.mean(np.abs(results['energy'][sind] - OCP * natoms))  # MAE
```

```{code-cell} ipython3
np.min(results['energy'][sind] - OCP * natoms), np.max(results['energy'][sind] - OCP * natoms)
```

```{code-cell} ipython3
plt.hist(results['energy'][sind] - OCP * natoms, bins=20);
```

Here we see many of the differences are very small. 0.0078125 = 1 / 128, and these errors strongly suggest some kind of mixed precision is responsible for these differences. It is an open issue to remove them and identify where the cause is.

```{code-cell} ipython3
(results['energy'][sind] - OCP * natoms)[0:400]
```
