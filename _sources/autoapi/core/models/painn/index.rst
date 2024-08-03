core.models.painn
=================

.. py:module:: core.models.painn


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/models/painn/painn/index
   /autoapi/core/models/painn/utils/index


Classes
-------

.. autoapisummary::

   core.models.painn.PaiNN


Package Contents
----------------

.. py:class:: PaiNN(hidden_channels: int = 512, num_layers: int = 6, num_rbf: int = 128, cutoff: float = 12.0, max_neighbors: int = 50, rbf: dict[str, str] | None = None, envelope: dict[str, str | int] | None = None, regress_forces: bool = True, direct_forces: bool = True, use_pbc: bool = True, otf_graph: bool = True, num_elements: int = 83, scale_file: str | None = None)

   Bases: :py:obj:`torch.nn.Module`, :py:obj:`fairchem.core.models.base.GraphModelMixin`


   PaiNN model based on the description in Schütt et al. (2021):
   Equivariant message passing for the prediction of tensorial properties
   and molecular spectra, https://arxiv.org/abs/2102.03150.


   .. py:attribute:: hidden_channels


   .. py:attribute:: num_layers


   .. py:attribute:: num_rbf


   .. py:attribute:: cutoff


   .. py:attribute:: max_neighbors


   .. py:attribute:: regress_forces


   .. py:attribute:: direct_forces


   .. py:attribute:: otf_graph


   .. py:attribute:: use_pbc


   .. py:attribute:: symmetric_edge_symmetrization
      :value: False



   .. py:attribute:: atom_emb


   .. py:attribute:: radial_basis


   .. py:attribute:: message_layers


   .. py:attribute:: update_layers


   .. py:attribute:: out_energy


   .. py:attribute:: inv_sqrt_2


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


   .. py:property:: num_params
      :type: int



   .. py:method:: __repr__() -> str

      Return repr(self).



