# Frequently Asked Questions

If you don't find your question answered here, please feel free to [file a GitHub issue](https://github.com/open-catalyst-project/ocp/issues) or [post on the discussion board](https://discuss.opencatalystproject.org/).

## Models

### Are predictions from OCP models deterministic?

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
where available (for example, see [`scatter_det`](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/common/utils.py#L1112)). Note that deterministic operations are often slower
than non-deterministic operations, so while this may be worth using for testing
and debugging, this is not recommended for large-scale training and inference.

### How do I train a model on OC20 total energies?

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
here](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md#oc20-reference-information).

To test if your setup is correct, try the following:

```python
from ocpmodels.datasets import OC22LmdbDataset

dset = OC22LmdbDataset({
    "src": "path/to/oc20/lmdb/folder/",
    "train_on_oc20_total_energies": True,
    "oc20_ref": "path/to/oc20_ref.pkl",
})

print(dset[0])
# Data(y=-181.54722937, ...) -- total DFT energies are usually quite high!
```

Another option that might be useful for training on total energies is passing
precomputed per-element average energies with [`lin_ref`](https://github.com/Open-Catalyst-Project/ocp/blob/faq/configs/s2ef/example.yml#L94-L97). If you use this option, make sure to recompute the
[normalizer statistics (for energies)](https://github.com/Open-Catalyst-Project/ocp/blob/faq/configs/s2ef/example.yml#L82-L83)
_after_ linear referencing.

### I'm trying to run GemNet-OC / GemNet-dT, but it throws an error that scaling factors are not fitted. What should I do?

GemNet-OC and GemNet-dT make use of empirical scaling factors that are fit on a
few batches of data prior to training in order to stabilize the variance of
activations. See [Sec. 6 in the GemNet paper](https://arxiv.org/abs/2106.08903)
for more details on this.

We provide some set of scaling factors as part of the `ocp` codebase that you
can reuse by passing the `scale_file` parameter in the YAML config. For example:
* GemNet-dT [scaling factors](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/gemnet/scaling_factors/gemnet-dT.json) and [config](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/gemnet/gemnet-dT.yml#L32)
* GemNet-OC [scaling factors](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc.pt) and [config](https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/gemnet/gemnet-oc.yml#L45)

If you change any of the model architecture hyperparameters or the dataset, you
should refit these scaling factors:

```bash
python ocpmodels/modules/scaling/fit.py \
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

### I'm trying to run GemNet-OC on my data, but it errors out on `sph_basis = self.spherical_basis(cosφ_cab, θ_cabd)`.

This is likely a tagging issue -- GemNet-OC computes quadruplet interactions
for atoms tagged as 1 and 2
([see code](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/gemnet_oc/gemnet_oc.py#L1020)).
In OC20 parlance, `tag==1` refers to surface atoms and `tag==2` refers to
adsorbate atoms. If all the atoms are tagged as 0 (check `atoms.get_tags()`),
no quadruplets are computed, and part of the GemNet-OC forward pass fails.
Having some atoms tagged as 1 or 2 in your structure should fix it.
