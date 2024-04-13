:py:mod:`ocpmodels.models.schnet`
=================================

.. py:module:: ocpmodels.models.schnet

.. autoapi-nested-parse::

   Copyright (c) Facebook, Inc. and its affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.schnet.SchNetWrap




.. py:class:: SchNetWrap(num_atoms: int, bond_feat_dim: int, num_targets: int, use_pbc: bool = True, regress_forces: bool = True, otf_graph: bool = False, hidden_channels: int = 128, num_filters: int = 128, num_interactions: int = 6, num_gaussians: int = 50, cutoff: float = 10.0, readout: str = 'add')


   Bases: :py:obj:`torch_geometric.nn.SchNet`, :py:obj:`ocpmodels.models.base.BaseModel`

   Wrapper around the continuous-filter convolutional neural network SchNet from the
   `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
   Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
   block of the form:

   .. math::
       \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
       h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

   :param num_atoms: Unused argument
   :type num_atoms: int
   :param bond_feat_dim: Unused argument
   :type bond_feat_dim: int
   :param num_targets: Number of targets to predict.
   :type num_targets: int
   :param use_pbc: If set to :obj:`True`, account for periodic boundary conditions.
                   (default: :obj:`True`)
   :type use_pbc: bool, optional
   :param regress_forces: If set to :obj:`True`, predict forces by differentiating
                          energy with respect to positions.
                          (default: :obj:`True`)
   :type regress_forces: bool, optional
   :param otf_graph: If set to :obj:`True`, compute graph edges on the fly.
                     (default: :obj:`False`)
   :type otf_graph: bool, optional
   :param hidden_channels: Number of hidden channels.
                           (default: :obj:`128`)
   :type hidden_channels: int, optional
   :param num_filters: Number of filters to use.
                       (default: :obj:`128`)
   :type num_filters: int, optional
   :param num_interactions: Number of interaction blocks
                            (default: :obj:`6`)
   :type num_interactions: int, optional
   :param num_gaussians: The number of gaussians :math:`\mu`.
                         (default: :obj:`50`)
   :type num_gaussians: int, optional
   :param cutoff: Cutoff distance for interatomic interactions.
                  (default: :obj:`10.0`)
   :type cutoff: float, optional
   :param readout: Whether to apply :obj:`"add"` or
                   :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
   :type readout: string, optional

   .. py:property:: num_params
      :type: int


   .. py:method:: _forward(data)


   .. py:method:: forward(data)

      :param z: Atomic number of each atom with shape
                :obj:`[num_atoms]`.
      :type z: torch.Tensor
      :param pos: Coordinates of each atom with shape
                  :obj:`[num_atoms, 3]`.
      :type pos: torch.Tensor
      :param batch: Batch indices assigning each atom
                    to a separate molecule with shape :obj:`[num_atoms]`.
                    (default: :obj:`None`)
      :type batch: torch.Tensor, optional



