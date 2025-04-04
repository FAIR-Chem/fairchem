# Model FAQ

If you don't find your question answered here, please feel free to [file a GitHub issue](https://github.com/FAIR-Chem/fairchem/issues).

## Are predictions from FAIRChem models deterministic?

By deterministic, we mean that multiple calls to the same function, given
the same inputs (and seed), will produce the same results.

On CPU, all operations should be deterministic. On GPU, `scatter` calls -- which
are used in the node aggregation functions to get the final energy --
are non-deterministic, since the order of parallel operations is not uniquely
determined [[1](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html),
[2](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html)].
Moreover, results may be different between GPU and CPU
executions [[3](https://pytorch.org/docs/stable/notes/randomness.html)].

To get deterministic results on GPU, use [`torch.use_deterministic_algorithms`](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms)
where available (for example, see [`scatter_det`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/common/utils.py#L1112)). Note that deterministic operations are often slower
than non-deterministic operations, so while this may be worth using for testing
and debugging, this is not recommended for large-scale training and inference.

## How do I train a model on OC20 total energies?

By default, the OC20 S2EF/IS2RE LMDBs have adsorption energies, i.e. the DFT
total energies minus the clean surface and gas phase adsorbate energies.

In order to train a model on DFT total energies, set the following flags in the
YAML config:

```yaml
task:
    ...
    # To train on OC20 total energies, use the 'oc22_lmdb' dataset class.
    dataset: oc22_lmdb
    ...

dataset:
    train:
        ...
        # To train on OC20 total energies, a path to OC20 reference energies
        # `oc20_ref` must be specified to unreference existing data.
        train_on_oc20_total_energies: True
        oc20_ref: path/to/oc20_ref.pkl
        ...
    val:
        ...
        train_on_oc20_total_energies: True
        oc20_ref: path/to/oc20_ref.pkl
        ...
```

The OC20 reference pickle file containing the energy necessary to convert
adsorption energy values to total energy is [available for download
here](https://fair-chem.github.io/core/datasets/oc20.html).

To test if your setup is correct, try the following:

```python
from fairchem.core.datasets import OC22LmdbDataset

dset = OC22LmdbDataset(
    {
        "src": "path/to/oc20/lmdb/folder/",
        "train_on_oc20_total_energies": True,
        "oc20_ref": "path/to/oc20_ref.pkl",
    }
)

print(dset[0])
# Data(y=-181.54722937, ...) -- total DFT energies are usually quite high!
```

Another option that might be useful for training on total energies is passing
precomputed per-element average energies with [`lin_ref`](https://github.com/FAIR-Chem/fairchem/blob/faq/configs/oc20/s2ef/example.yml#L94-L97). If you use this option, make sure to recompute the
[normalizer statistics (for energies)](https://github.com/FAIR-Chem/fairchem/blob/faq/configs/oc20/s2ef/example.yml#L82-L83)
_after_ linear referencing.

## I'm trying to run GemNet-OC / GemNet-dT, but it throws an error that scaling factors are not fitted. What should I do?

GemNet-OC and GemNet-dT make use of empirical scaling factors that are fit on a
few batches of data prior to training in order to stabilize the variance of
activations. See [Sec. 6 in the GemNet paper](https://arxiv.org/abs/2106.08903)
for more details on this.

We provide some set of scaling factors as part of the `fairchem` codebase that you
can reuse by passing the `scale_file` parameter in the YAML config. For example:
* GemNet-dT [scaling factors](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/configs/oc20/s2ef/all/gemnet/scaling_factors/gemnet-dT.json) and [config](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/configs/oc20/s2ef/all/gemnet/gemnet-dT.yml#L32)
* GemNet-OC [scaling factors](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/configs/oc20/s2ef/all/gemnet/scaling_factors/gemnet-oc.pt) and [config](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/configs/oc20/s2ef/all/gemnet/gemnet-oc.yml#L45)

If you change any of the model architecture hyperparameters or the dataset, you
should refit these scaling factors:

```bash
python src/fairchem/core/modules/scaling/fit.py \
    --config-yml path/to/my/config.yml \
    --checkpoint path/to/save/checkpoint.pt \
    --mode train
```

This will recalculate the scaling factors and save them in a checkpoint file
`path/to/save/checkpoint.pt`, that you can then load and launch training from:

```bash
python main.py \
    --config-yml path/to/my/config.yml \
    --checkpoint path/to/save/checkpoint.pt \
    --mode train
```

## I'm trying to run GemNet-OC on my data, but it errors out on `sph_basis = self.spherical_basis(cosφ_cab, θ_cabd)`.

This is likely a tagging issue -- GemNet-OC computes quadruplet interactions
for atoms tagged as 1 and 2
([see code](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet_oc/gemnet_oc.py#L1020)).
In OC20 parlance, `tag==1` refers to surface atoms and `tag==2` refers to
adsorbate atoms. If all the atoms are tagged as 0 (check `atoms.get_tags()`),
no quadruplets are computed, and part of the GemNet-OC forward pass fails.
Having some atoms tagged as 1 or 2 in your structure should fix it.



## OutOfMemoryError

If you see errors like:

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 390.00 MiB (GPU 0; 10.76 GiB total capacity; 9.59 GiB already allocated; 170.06 MiB free; 9.81 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

It means your GPU is out of memory. Some reasons could be that you have multiple notebooks open that are using the GPU, e.g. they have loaded a calculator or something. Try closing all the other notebooks.

It could also mean the batch size is too large to fit in memory. You can try making it smaller in the yml config file (optim.batch_size).

It is recommended you use automatic mixed precision, --amp, in the options to main.py, or in the config.yml.

If it is an option, you can try a GPU with more memory, or you may be able to split the job over multiple GPUs.

+++

## I want the energy of a gas phase atom

But I get an error like

```
RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous
```

The problem here is that no neighbors are found for the single atom which causes an error. This may be model dependent. There is currently no way to get atomic energies for some models.

```{code-cell} ipython3
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file
checkpoint_path = model_name_to_local_file('GemNet-OC-S2EFS-OC20+OC22', local_cache='/tmp/fairchem_checkpoints/')
calc = OCPCalculator(checkpoint_path=checkpoint_path)
```

```{code-cell} ipython3
%%capture
from ase.build import bulk
atoms = bulk('Cu', a=10)
atoms.set_calculator(calc)
atoms.get_potential_energy()
```

# I get wildly different energies from the different models

Some models are trained on adsorption energies, and some are trained on total energies. You have to know which one you are using.

Sometimes you can tell by the magnitude of energies, but you should use care with this. If energies are "small" and near zero they are likely adsorption energies. If energies are "large" in magnitude they are probably total energies. This can be misleading though, as it depends on the total number of atoms in the systems.

```{code-cell} ipython3
# These are to suppress the output from making the calculators.
from io import StringIO
import contextlib
```

```{code-cell} ipython3
from ase.build import fcc111, add_adsorbate
slab = fcc111('Pt', size=(2, 2, 5), vacuum=10.0)
add_adsorbate(slab, 'O', height=1.2, position='fcc')
```

```{code-cell} ipython3
from fairchem.core.models.model_registry import model_name_to_local_file

# OC20 model - trained on adsorption energies
checkpoint_path = model_name_to_local_file('GemNet-OC-S2EF-OC20-All', local_cache='/tmp/fairchem_checkpoints/')

with contextlib.redirect_stdout(StringIO()) as _:
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)



slab.set_calculator(calc)
slab.get_potential_energy()
```

```{code-cell} ipython3
# An OC22 checkpoint - trained on total energy
checkpoint_path = model_name_to_local_file('GemNet-OC-S2EFS-OC20+OC22', local_cache='/tmp/fairchem_checkpoints/')

with contextlib.redirect_stdout(StringIO()) as _:
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)



slab.set_calculator(calc)
slab.get_potential_energy()
```

```{code-cell} ipython3
# This eSCN model is trained on adsorption energies
checkpoint_path = model_name_to_local_file('eSCN-L4-M2-Lay12-S2EF-OC20-2M', local_cache='/tmp/fairchem_checkpoints/')

with contextlib.redirect_stdout(StringIO()) as _:
    calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)

slab.set_calculator(calc)
slab.get_potential_energy()
```

# Miscellaneous warnings

In general, warnings are not errors.

## Unrecognized arguments

With Gemnet models you might see warnings like:

```
WARNING:root:Unrecognized arguments: ['symmetric_edge_symmetrization']
```

You can ignore this warning, it is not important for predictions.

## Unable to identify ocp trainer

The trainer is not specified in some checkpoints, and defaults to `forces` which means energy and forces are calculated. This is the default for the ASE OCP calculator, and this warning just alerts you it is setting that.

```
WARNING:root:Unable to identify ocp trainer, defaulting to `forces`. Specify the `trainer` argument into OCPCalculator if otherwise.
```

+++

## Request entity too large - can't save your Notebook

If you run commands that generate a lot of output in a notebook, sometimes the Jupyter notebook will become too large to save. It is kind of sad, the only thing I know to do is delete the output of the cell. Then maybe you can save it.

A solution after you know it happens is redirect output to a file.

This has happened when running training in a notebook where there are too many lines of output, or if you have a lot (20+) of inline images.

+++

## You need at least four atoms for molecules with some models

Gemnet in particular seems to require at least 4 atoms. This has to do with interactions between atoms and their neighbors.

```{code-cell} ipython3
%%capture
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file
import os

checkpoint_path = model_name_to_local_file('GemNet-OC-S2EFS-OC20+OC22', local_cache='/tmp/fairchem_checkpoints/')

calc = OCPCalculator(checkpoint_path=checkpoint_path)
```

```{code-cell} ipython3
%%capture
from ase.build import molecule
import numpy as np

atoms = molecule('H2O')
atoms.set_tags(np.ones(len(atoms)))
atoms.set_calculator(calc)
atoms.get_potential_energy()
```

## To tag or not?

Some models use tags to determine which atoms to calculate energies for. For example, Gemnet uses a tag=1 to indicate the atom should be calculated. You will get an error with this model

```{code-cell} ipython3
%%capture
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file
import os

checkpoint_path = model_name_to_local_file('GemNet-OC-S2EFS-OC20+OC22', local_cache='/tmp/fairchem_checkpoints/')
calc = OCPCalculator(checkpoint_path=checkpoint_path)
```

```{code-cell} ipython3
%%capture
atoms = molecule('CH4')
atoms.set_calculator(calc)
atoms.get_potential_energy()  # error
```

```{code-cell} ipython3
atoms = molecule('CH4')
atoms.set_tags(np.ones(len(atoms)))  # <- critical line for Gemnet
atoms.set_calculator(calc)
atoms.get_potential_energy()
```

Not all models require tags though. This EquiformerV2 model does not use them. This is another detail that is important to keep in mind.

```{code-cell} ipython3
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file
import os

checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/fairchem_checkpoints/')

calc = OCPCalculator(checkpoint_path=checkpoint_path)
```

```{code-cell} ipython3
atoms = molecule('CH4')

atoms.set_calculator(calc)
atoms.get_potential_energy()
```

## Stochastic simulation results

Some models are not deterministic (SCN/eSCN/EqV2), i.e. you can get slightly different answers each time you run it.
An example is shown below. See [Issue 563](https://github.com/FAIR-Chem/fairchem/issues/563) for more discussion.
This happens because a random selection of is made to sample edges, and a different selection is made each time you run it.

```{code-cell} ipython3
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator

checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/fairchem_checkpoints/')
calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)

from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
slab = fcc111('Pt', size=(2, 2, 5), vacuum=10.0)
add_adsorbate(slab, 'O', height=1.2, position='fcc')
slab.set_calculator(calc)

results = []
for i in range(10):
    calc.calculate(slab, ['energy'], None)
    results += [slab.get_potential_energy()]

import numpy as np
print(np.mean(results), np.std(results))
for result in results:
    print(result)
```


## The forces don't sum to zero

In DFT, the forces on all the atoms should sum to zero; otherwise, there is a net translational or rotational force present. This is not enforced in fairchem models. Instead, individual forces are predicted, with no constraint that they sum to zero. If the force predictions are very accurate, then they sum close to zero. You can further improve this if you subtract the mean force from each atom.

```{code-cell} ipython3
from fairchem.core.models.model_registry import model_name_to_local_file
checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/fairchem_checkpoints/')

from fairchem.core.common.relaxation.ase_utils import OCPCalculator
calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)

from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
slab = fcc111('Pt', size=(2, 2, 5), vacuum=10.0)
add_adsorbate(slab, 'O', height=1.2, position='fcc')
slab.set_calculator(calc)

f = slab.get_forces()
f.sum(axis=0)
```

```{code-cell} ipython3
# This makes them sum closer to zero by removing net translational force
(f - f.mean(axis=0)).sum(axis=0)
```
