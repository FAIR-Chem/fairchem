:py:mod:`ocpmodels.models.painn.painn`
======================================

.. py:module:: ocpmodels.models.painn.painn

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.

   ---

   MIT License

   Copyright (c) 2021 www.compscience.org

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.painn.painn.PaiNN
   ocpmodels.models.painn.painn.PaiNNMessage
   ocpmodels.models.painn.painn.PaiNNUpdate
   ocpmodels.models.painn.painn.PaiNNOutput
   ocpmodels.models.painn.painn.GatedEquivariantBlock




.. py:class:: PaiNN(num_atoms: int, bond_feat_dim: int, num_targets: int, hidden_channels: int = 512, num_layers: int = 6, num_rbf: int = 128, cutoff: float = 12.0, max_neighbors: int = 50, rbf: dict[str, str] | None = None, envelope: dict[str, str | int] | None = None, regress_forces: bool = True, direct_forces: bool = True, use_pbc: bool = True, otf_graph: bool = True, num_elements: int = 83, scale_file: str | None = None)


   Bases: :py:obj:`ocpmodels.models.base.BaseModel`

   PaiNN model based on the description in Schütt et al. (2021):
   Equivariant message passing for the prediction of tensorial properties
   and molecular spectra, https://arxiv.org/abs/2102.03150.

   .. py:property:: num_params
      :type: int


   .. py:method:: reset_parameters() -> None


   .. py:method:: select_symmetric_edges(tensor, mask, reorder_idx, inverse_neg) -> torch.Tensor


   .. py:method:: symmetrize_edges(edge_index, cell_offsets, neighbors, batch_idx, reorder_tensors, reorder_tensors_invneg)

      Symmetrize edges to ensure existence of counter-directional edges.

      Some edges are only present in one direction in the data,
      since every atom has a maximum number of neighbors.
      If `symmetric_edge_symmetrization` is False,
      we only use i->j edges here. So we lose some j->i edges
      and add others by making it symmetric.
      If `symmetric_edge_symmetrization` is True,
      we always use both directions.


   .. py:method:: generate_graph_values(data)


   .. py:method:: forward(data)


   .. py:method:: __repr__() -> str

      Return repr(self).



.. py:class:: PaiNNMessage(hidden_channels, num_rbf)


   Bases: :py:obj:`torch_geometric.nn.MessagePassing`

   Base class for creating message passing layers of the form

   .. math::
       \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
       \bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
       \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

   where :math:`\bigoplus` denotes a differentiable, permutation invariant
   function, *e.g.*, sum, mean, min, max or mul, and
   :math:`\gamma_{\mathbf{\Theta}}` and :math:`\phi_{\mathbf{\Theta}}` denote
   differentiable functions such as MLPs.
   See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
   create_gnn.html>`__ for the accompanying tutorial.

   :param aggr: The aggregation scheme
                to use, *e.g.*, :obj:`"add"`, :obj:`"sum"` :obj:`"mean"`,
                :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`.
                In addition, can be any
                :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
                that automatically resolves to it).
                If given as a list, will make use of multiple aggregations in which
                different outputs will get concatenated in the last dimension.
                If set to :obj:`None`, the :class:`MessagePassing` instantiation is
                expected to implement its own aggregation logic via
                :meth:`aggregate`. (default: :obj:`"add"`)
   :type aggr: str or [str] or Aggregation, optional
   :param aggr_kwargs: Arguments passed to the
                       respective aggregation function in case it gets automatically
                       resolved. (default: :obj:`None`)
   :type aggr_kwargs: Dict[str, Any], optional
   :param flow: The flow direction of message passing
                (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
                (default: :obj:`"source_to_target"`)
   :type flow: str, optional
   :param node_dim: The axis along which to propagate.
                    (default: :obj:`-2`)
   :type node_dim: int, optional
   :param decomposed_layers: The number of feature decomposition
                             layers, as introduced in the `"Optimizing Memory Efficiency of
                             Graph Neural Networks on Edge Computing Platforms"
                             <https://arxiv.org/abs/2104.03058>`_ paper.
                             Feature decomposition reduces the peak memory usage by slicing
                             the feature dimensions into separated feature decomposition layers
                             during GNN aggregation.
                             This method can accelerate GNN execution on CPU-based platforms
                             (*e.g.*, 2-3x speedup on the
                             :class:`~torch_geometric.datasets.Reddit` dataset) for common GNN
                             models such as :class:`~torch_geometric.nn.models.GCN`,
                             :class:`~torch_geometric.nn.models.GraphSAGE`,
                             :class:`~torch_geometric.nn.models.GIN`, etc.
                             However, this method is not applicable to all GNN operators
                             available, in particular for operators in which message computation
                             can not easily be decomposed, *e.g.* in attention-based GNNs.
                             The selection of the optimal value of :obj:`decomposed_layers`
                             depends both on the specific graph dataset and available hardware
                             resources.
                             A value of :obj:`2` is suitable in most cases.
                             Although the peak memory usage is directly associated with the
                             granularity of feature decomposition, the same is not necessarily
                             true for execution speedups. (default: :obj:`1`)
   :type decomposed_layers: int, optional

   .. py:method:: reset_parameters() -> None

      Resets all learnable parameters of the module.


   .. py:method:: forward(x, vec, edge_index, edge_rbf, edge_vector)

      Runs the forward pass of the module.


   .. py:method:: message(xh_j, vec_j, rbfh_ij, r_ij)

      Constructs messages from node :math:`j` to node :math:`i`
      in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
      :obj:`edge_index`.
      This function can take any argument as input which was initially
      passed to :meth:`propagate`.
      Furthermore, tensors passed to :meth:`propagate` can be mapped to the
      respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
      :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.


   .. py:method:: aggregate(features: tuple[torch.Tensor, torch.Tensor], index: torch.Tensor, dim_size: int) -> tuple[torch.Tensor, torch.Tensor]

      Aggregates messages from neighbors as
      :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

      Takes in the output of message computation as first argument and any
      argument which was initially passed to :meth:`propagate`.

      By default, this function will delegate its call to the underlying
      :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
      as specified in :meth:`__init__` by the :obj:`aggr` argument.


   .. py:method:: update(inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]

      Updates node embeddings in analogy to
      :math:`\gamma_{\mathbf{\Theta}}` for each node
      :math:`i \in \mathcal{V}`.
      Takes in the output of aggregation as first argument and any argument
      which was initially passed to :meth:`propagate`.



.. py:class:: PaiNNUpdate(hidden_channels)


   Bases: :py:obj:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:method:: reset_parameters() -> None


   .. py:method:: forward(x, vec)



.. py:class:: PaiNNOutput(hidden_channels)


   Bases: :py:obj:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:method:: reset_parameters() -> None


   .. py:method:: forward(x, vec)



.. py:class:: GatedEquivariantBlock(hidden_channels, out_channels)


   Bases: :py:obj:`torch.nn.Module`

   Gated Equivariant Block as defined in Schütt et al. (2021):
   Equivariant message passing for the prediction of tensorial properties and molecular spectra

   .. py:method:: reset_parameters() -> None


   .. py:method:: forward(x, v)



