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

Common gotchas with OCP
---------------------------------

```{code-cell} ipython3
%run ocp-tutorial.ipynb
```

# OutOfMemoryError

If you see errors like:

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 390.00 MiB (GPU 0; 10.76 GiB total capacity; 9.59 GiB already allocated; 170.06 MiB free; 9.81 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

It means your GPU is out of memory. Some reasons could be that you have multiple notebooks open that are using the GPU, e.g. they have loaded a calculator or something. Try closing all the other notebooks. 

It could also mean the batch size is too large to fit in memory. You can try making it smaller in the yml config file (optim.batch_size).

It is recommended you use automatic mixed precision, --amp, in the options to main.py, or in the config.yml.

If it is an option, you can try a GPU with more memory, or you may be able to split the job over multiple GPUs.

+++

# I want the energy of a gas phase atom

But I get an error like 

```
RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous
```

The problem here is that no neighbors are found for the single atom which causes an error. This may be model dependent. There is currently no way to get atomic energies for some models.

```{code-cell} ipython3
%%capture
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
cp = "gnoc_oc22_oc20_all_s2ef.pt"
calc = OCPCalculator(checkpoint=cp)
```

```{code-cell} ipython3
from ase.build import bulk
atoms = bulk('Cu', a=10)
atoms.set_calculator(calc)
atoms.get_potential_energy()
```

# I get wildly different energies from the different models

Some models are trained on adsorption energies, and some are trained on total energies. You have to know which one you are using.

Sometimes you can tell by the magnitude of energies, but you should use care with this. If energies are "small" and near zero they are likely adsorption energies. If energies are "large" in magnitude they are probably total energies. This can be misleading though, as it depends on the total number of atoms in the systems.

```{code-cell} ipython3
%run ocp-tutorial.ipynb

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
# OC20 model - trained on adsorption energies
checkpoint = get_checkpoint('GemNet-OC All')

with contextlib.redirect_stdout(StringIO()) as _:
    calc = OCPCalculator(checkpoint=os.path.expanduser(checkpoint), cpu=False)
    


slab.set_calculator(calc)
slab.get_potential_energy()
```

```{code-cell} ipython3
# An OC22 checkpoint - trained on total energy
checkpoint = get_checkpoint('GemNet-OC OC22')

with contextlib.redirect_stdout(StringIO()) as _:
    calc = OCPCalculator(checkpoint=checkpoint, cpu=False)
    


slab.set_calculator(calc)
slab.get_potential_energy()
```

```{code-cell} ipython3
# This eSCN model is trained on adsorption energies
checkpoint = get_checkpoint('eSCN-L4-M2-Lay12 2M')

with contextlib.redirect_stdout(StringIO()) as _:
    calc = OCPCalculator(checkpoint=checkpoint, cpu=False)

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

## Unable to identify OCP trainer

The trainer is not specified in some checkpoints, and defaults to `forces` which means energy and forces are calculated. This is the default for the ASE OCP calculator, and this warning just alerts you it is setting that.

```
WARNING:root:Unable to identify OCP trainer, defaulting to `forces`. Specify the `trainer` argument into OCPCalculator if otherwise.
```

+++

# Request entity too large - can't save your Notebook

If you run commands that generate a lot of output in a notebook, sometimes the Jupyter notebook will become too large to save. It is kind of sad, the only thing I know to do is delete the output of the cell. Then maybe you can save it.

A solution after you know it happens is redirect output to a file.

This has happened when running training in a notebook where there are too many lines of output, or if you have a lot (20+) of inline images.

+++

# You need at least four atoms for molecules with some models

Gemnet in particular seems to require at least 4 atoms. This has to do with interactions between atoms and their neighbors.

```{code-cell} ipython3
%%capture
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import os
cp = checkpoint = get_checkpoint('GemNet-OC OC20+OC22')
calc = OCPCalculator(checkpoint=cp)
```

```{code-cell} ipython3
from ase.build import molecule
atoms = molecule('H2O')
atoms.set_tags(np.ones(len(atoms)))
atoms.set_calculator(calc)
atoms.get_potential_energy()
```

# To tag or not?

Some models use tags to determine which atoms to calculate energies for. For example, Gemnet uses a tag=1 to indicate the atom should be calculated. You will get an error with this model 

```{code-cell} ipython3
%%capture
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import os
cp = checkpoint = get_checkpoint('GemNet-OC OC20+OC22')
calc = OCPCalculator(checkpoint=cp)
```

```{code-cell} ipython3
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

Not all models require tags though. This eSCN model does not use them. This is another detail that is important to keep in mind.

```{code-cell} ipython3
%%capture
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import os
cp = checkpoint = get_checkpoint('eSCN-L6-M3-Lay20 All+MD')
calc = OCPCalculator(checkpoint=cp)
```

```{code-cell} ipython3
atoms = molecule('CH4')

atoms.set_calculator(calc)
atoms.get_potential_energy()
```

# Stochastic simulation results

Some models are not deterministic (SCN/eSCN.EqV2), i.e. you can get slightly different answers each time you run it. An example is shown below. See https://github.com/Open-Catalyst-Project/ocp/issues/563 for more discussion. This happens because a random selection of is made to sample edges, and a different selection is made each time you run it. 

```{code-cell} ipython3
%run ocp-tutorial.ipynb
checkpoint = get_checkpoint('eSCN-L6-M3-Lay20 All+MD')

from ocpmodels.common.relaxation.ase_utils import OCPCalculator
calc = OCPCalculator(checkpoint=os.path.expanduser(checkpoint), cpu=True)

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

```{code-cell} ipython3
%run ocp-tutorial.ipynb

import os
from ocpmodels.common.relaxation.ase_utils import OCPCalculator

for ckp in checkpoints:
    try:
        checkpoint = get_checkpoint(ckp)
        calc = OCPCalculator(checkpoint, cpu=True)
    except Exception as exc:
        print(ckp, exc)
    finally:
        os.unlink(checkpoint)
```

# The forces don't sum to zero

In DFT, the forces on all the atoms should sum to zero; otherwise, there is a net translational or rotational force present. This is not enforced in OCP models. Instead, individual forces are predicted, with no constraint that they sum to zero. If the force predictions are very accurate, then they sum close to zero. You can further improve this if you subtract the mean force from each atom.

```{code-cell} ipython3
%run ocp-tutorial.ipynb
checkpoint = get_checkpoint('eSCN-L6-M3-Lay20 All+MD')

from ocpmodels.common.relaxation.ase_utils import OCPCalculator
calc = OCPCalculator(checkpoint=os.path.expanduser(checkpoint), cpu=True)

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
