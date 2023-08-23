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
