:py:mod:`core.models.painn`
===========================

.. py:module:: core.models.painn


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   painn/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   core.models.painn.PaiNN




.. py:class:: PaiNN(num_atoms: int, bond_feat_dim: int, num_targets: int, hidden_channels: int = 512, num_layers: int = 6, num_rbf: int = 128, cutoff: float = 12.0, max_neighbors: int = 50, rbf: dict[str, str] | None = None, envelope: dict[str, str | int] | None = None, regress_forces: bool = True, direct_forces: bool = True, use_pbc: bool = True, otf_graph: bool = True, num_elements: int = 83, scale_file: str | None = None)


   Bases: :py:obj:`fairchem.core.models.base.BaseModel`

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



