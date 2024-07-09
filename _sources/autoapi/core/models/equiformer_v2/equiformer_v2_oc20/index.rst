core.models.equiformer_v2.equiformer_v2_oc20
============================================

.. py:module:: core.models.equiformer_v2.equiformer_v2_oc20


Attributes
----------

.. autoapisummary::

   core.models.equiformer_v2.equiformer_v2_oc20._AVG_NUM_NODES
   core.models.equiformer_v2.equiformer_v2_oc20._AVG_DEGREE


Classes
-------

.. autoapisummary::

   core.models.equiformer_v2.equiformer_v2_oc20.EquiformerV2_OC20


Module Contents
---------------

.. py:data:: _AVG_NUM_NODES
   :value: 77.81317


.. py:data:: _AVG_DEGREE
   :value: 23.395238876342773


.. py:class:: EquiformerV2_OC20(num_atoms: int, bond_feat_dim: int, num_targets: int, use_pbc: bool = True, regress_forces: bool = True, otf_graph: bool = True, max_neighbors: int = 500, max_radius: float = 5.0, max_num_elements: int = 90, num_layers: int = 12, sphere_channels: int = 128, attn_hidden_channels: int = 128, num_heads: int = 8, attn_alpha_channels: int = 32, attn_value_channels: int = 16, ffn_hidden_channels: int = 512, norm_type: str = 'rms_norm_sh', lmax_list: list[int] | None = None, mmax_list: list[int] | None = None, grid_resolution: int | None = None, num_sphere_samples: int = 128, edge_channels: int = 128, use_atom_edge_embedding: bool = True, share_atom_edge_embedding: bool = False, use_m_share_rad: bool = False, distance_function: str = 'gaussian', num_distance_basis: int = 512, attn_activation: str = 'scaled_silu', use_s2_act_attn: bool = False, use_attn_renorm: bool = True, ffn_activation: str = 'scaled_silu', use_gate_act: bool = False, use_grid_mlp: bool = False, use_sep_s2_act: bool = True, alpha_drop: float = 0.1, drop_path_rate: float = 0.05, proj_drop: float = 0.0, weight_init: str = 'normal', enforce_max_neighbors_strictly: bool = True, avg_num_nodes: float | None = None, avg_degree: float | None = None, use_energy_lin_ref: bool | None = False, load_energy_lin_ref: bool | None = False)

   Bases: :py:obj:`fairchem.core.models.base.BaseModel`


   Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

   :param use_pbc: Use periodic boundary conditions
   :type use_pbc: bool
   :param regress_forces: Compute forces
   :type regress_forces: bool
   :param otf_graph: Compute graph On The Fly (OTF)
   :type otf_graph: bool
   :param max_neighbors: Maximum number of neighbors per atom
   :type max_neighbors: int
   :param max_radius: Maximum distance between nieghboring atoms in Angstroms
   :type max_radius: float
   :param max_num_elements: Maximum atomic number
   :type max_num_elements: int
   :param num_layers: Number of layers in the GNN
   :type num_layers: int
   :param sphere_channels: Number of spherical channels (one set per resolution)
   :type sphere_channels: int
   :param attn_hidden_channels: Number of hidden channels used during SO(2) graph attention
   :type attn_hidden_channels: int
   :param num_heads: Number of attention heads
   :type num_heads: int
   :param attn_alpha_head: Number of channels for alpha vector in each attention head
   :type attn_alpha_head: int
   :param attn_value_head: Number of channels for value vector in each attention head
   :type attn_value_head: int
   :param ffn_hidden_channels: Number of hidden channels used during feedforward network
   :type ffn_hidden_channels: int
   :param norm_type: Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])
   :type norm_type: str
   :param lmax_list: List of maximum degree of the spherical harmonics (1 to 10)
   :type lmax_list: int
   :param mmax_list: List of maximum order of the spherical harmonics (0 to lmax)
   :type mmax_list: int
   :param grid_resolution: Resolution of SO3_Grid
   :type grid_resolution: int
   :param num_sphere_samples: Number of samples used to approximate the integration of the sphere in the output blocks
   :type num_sphere_samples: int
   :param edge_channels: Number of channels for the edge invariant features
   :type edge_channels: int
   :param use_atom_edge_embedding: Whether to use atomic embedding along with relative distance for edge scalar features
   :type use_atom_edge_embedding: bool
   :param share_atom_edge_embedding: Whether to share `atom_edge_embedding` across all blocks
   :type share_atom_edge_embedding: bool
   :param use_m_share_rad: Whether all m components within a type-L vector of one channel share radial function weights
   :type use_m_share_rad: bool
   :param distance_function: Basis function used for distances
   :type distance_function: "gaussian", "sigmoid", "linearsigmoid", "silu"
   :param attn_activation: Type of activation function for SO(2) graph attention
   :type attn_activation: str
   :param use_s2_act_attn: Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
   :type use_s2_act_attn: bool
   :param use_attn_renorm: Whether to re-normalize attention weights
   :type use_attn_renorm: bool
   :param ffn_activation: Type of activation function for feedforward network
   :type ffn_activation: str
   :param use_gate_act: If `True`, use gate activation. Otherwise, use S2 activation
   :type use_gate_act: bool
   :param use_grid_mlp: If `True`, use projecting to grids and performing MLPs for FFNs.
   :type use_grid_mlp: bool
   :param use_sep_s2_act: If `True`, use separable S2 activation when `use_gate_act` is False.
   :type use_sep_s2_act: bool
   :param alpha_drop: Dropout rate for attention weights
   :type alpha_drop: float
   :param drop_path_rate: Drop path rate
   :type drop_path_rate: float
   :param proj_drop: Dropout rate for outputs of attention and FFN in Transformer blocks
   :type proj_drop: float
   :param weight_init: ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
   :type weight_init: str
   :param enforce_max_neighbors_strictly: When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
   :type enforce_max_neighbors_strictly: bool
   :param avg_num_nodes: Average number of nodes per graph
   :type avg_num_nodes: float
   :param avg_degree: Average degree of nodes in the graph
   :type avg_degree: float
   :param use_energy_lin_ref: Whether to add the per-atom energy references during prediction.
                              During training and validation, this should be kept `False` since we use the `lin_ref` parameter in the OC22 dataloader to subtract the per-atom linear references from the energy targets.
                              During prediction (where we don't have energy targets), this can be set to `True` to add the per-atom linear references to the predicted energies.
   :type use_energy_lin_ref: bool
   :param load_energy_lin_ref: Whether to add nn.Parameters for the per-element energy references.
                               This additional flag is there to ensure compatibility when strict-loading checkpoints, since the `use_energy_lin_ref` flag can be either True or False even if the model is trained with linear references.
                               You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine.
   :type load_energy_lin_ref: bool


   .. py:method:: _init_gp_partitions(atomic_numbers_full, data_batch_full, edge_index, edge_distance, edge_distance_vec)

      Graph Parallel
      This creates the required partial tensors for each rank given the full tensors.
      The tensors are split on the dimension along the node index using node_partition.



   .. py:method:: forward(data)


   .. py:method:: _init_edge_rot_mat(data, edge_index, edge_distance_vec)


   .. py:property:: num_params


   .. py:method:: _init_weights(m)


   .. py:method:: _uniform_init_rad_func_linear_weights(m)


   .. py:method:: _uniform_init_linear_weights(m)


   .. py:method:: no_weight_decay() -> set

      Returns a list of parameters with no weight decay.



