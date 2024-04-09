:py:mod:`ocpmodels.models.equiformer_v2.transformer_block`
==========================================================

.. py:module:: ocpmodels.models.equiformer_v2.transformer_block


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   ocpmodels.models.equiformer_v2.transformer_block.SO2EquivariantGraphAttention
   ocpmodels.models.equiformer_v2.transformer_block.FeedForwardNetwork
   ocpmodels.models.equiformer_v2.transformer_block.TransBlockV2




.. py:class:: SO2EquivariantGraphAttention(sphere_channels: int, hidden_channels: int, num_heads: int, attn_alpha_channels: int, attn_value_channels: int, output_channels: int, lmax_list: List[int], mmax_list: List[int], SO3_rotation, mappingReduced, SO3_grid, max_num_elements: int, edge_channels_list, use_atom_edge_embedding: bool = True, use_m_share_rad: bool = False, activation='scaled_silu', use_s2_act_attn: bool = False, use_attn_renorm: bool = True, use_gate_act: bool = False, use_sep_s2_act: bool = True, alpha_drop: float = 0.0)


   Bases: :py:obj:`torch.nn.Module`

   SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
       SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
       attention weights * non-linear messages -> Linear

   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param hidden_channels: Number of hidden channels used during the SO(2) conv
   :type hidden_channels: int
   :param num_heads: Number of attention heads
   :type num_heads: int
   :param attn_alpha_head: Number of channels for alpha vector in each attention head
   :type attn_alpha_head: int
   :param attn_value_head: Number of channels for value vector in each attention head
   :type attn_value_head: int
   :param output_channels: Number of output channels
   :type output_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution
   :param SO3_rotation (list: SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
   :param mappingReduced: Class to convert l and m indices once node embedding is rotated
   :type mappingReduced: CoefficientMappingModule
   :param SO3_grid: Class used to convert from grid the spherical harmonic representations
   :type SO3_grid: SO3_grid
   :param max_num_elements: Maximum number of atomic numbers
   :type max_num_elements: int
   :param edge_channels_list (list: int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                    The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
   :param use_atom_edge_embedding: Whether to use atomic embedding along with relative distance for edge scalar features
   :type use_atom_edge_embedding: bool
   :param use_m_share_rad: Whether all m components within a type-L vector of one channel share radial function weights
   :type use_m_share_rad: bool
   :param activation: Type of activation function
   :type activation: str
   :param use_s2_act_attn: Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
   :type use_s2_act_attn: bool
   :param use_attn_renorm: Whether to re-normalize attention weights
   :type use_attn_renorm: bool
   :param use_gate_act: If `True`, use gate activation. Otherwise, use S2 activation.
   :type use_gate_act: bool
   :param use_sep_s2_act: If `True`, use separable S2 activation when `use_gate_act` is False.
   :type use_sep_s2_act: bool
   :param alpha_drop: Dropout rate for attention weights
   :type alpha_drop: float

   .. py:method:: forward(x: torch.Tensor, atomic_numbers, edge_distance: torch.Tensor, edge_index)



.. py:class:: FeedForwardNetwork(sphere_channels: int, hidden_channels: int, output_channels: int, lmax_list: List[int], mmax_list: List[int], SO3_grid, activation: str = 'scaled_silu', use_gate_act: bool = False, use_grid_mlp: bool = False, use_sep_s2_act: bool = True)


   Bases: :py:obj:`torch.nn.Module`

   FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

   :param sphere_channels: Number of spherical channels
   :type sphere_channels: int
   :param hidden_channels: Number of hidden channels used during feedforward network
   :type hidden_channels: int
   :param output_channels: Number of output channels
   :type output_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution
   :param SO3_grid: Class used to convert from grid the spherical harmonic representations
   :type SO3_grid: SO3_grid
   :param activation: Type of activation function
   :type activation: str
   :param use_gate_act: If `True`, use gate activation. Otherwise, use S2 activation
   :type use_gate_act: bool
   :param use_grid_mlp: If `True`, use projecting to grids and performing MLPs.
   :type use_grid_mlp: bool
   :param use_sep_s2_act: If `True`, use separable grid MLP when `use_grid_mlp` is True.
   :type use_sep_s2_act: bool

   .. py:method:: forward(input_embedding)



.. py:class:: TransBlockV2(sphere_channels: int, attn_hidden_channels: int, num_heads: int, attn_alpha_channels: int, attn_value_channels: int, ffn_hidden_channels: int, output_channels: int, lmax_list: List[int], mmax_list: List[int], SO3_rotation, mappingReduced, SO3_grid, max_num_elements: int, edge_channels_list: List[int], use_atom_edge_embedding: bool = True, use_m_share_rad: bool = False, attn_activation: str = 'silu', use_s2_act_attn: bool = False, use_attn_renorm: bool = True, ffn_activation: str = 'silu', use_gate_act: bool = False, use_grid_mlp: bool = False, use_sep_s2_act: bool = True, norm_type: str = 'rms_norm_sh', alpha_drop: float = 0.0, drop_path_rate: float = 0.0, proj_drop: float = 0.0)


   Bases: :py:obj:`torch.nn.Module`

   :param sphere_channels: Number of spherical channels
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
   :param output_channels: Number of output channels
   :type output_channels: int
   :param lmax_list (list: int):       List of degrees (l) for each resolution
   :param mmax_list (list: int):       List of orders (m) for each resolution
   :param SO3_rotation (list: SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
   :param mappingReduced: Class to convert l and m indices once node embedding is rotated
   :type mappingReduced: CoefficientMappingModule
   :param SO3_grid: Class used to convert from grid the spherical harmonic representations
   :type SO3_grid: SO3_grid
   :param max_num_elements: Maximum number of atomic numbers
   :type max_num_elements: int
   :param edge_channels_list (list: int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                    The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
   :param use_atom_edge_embedding: Whether to use atomic embedding along with relative distance for edge scalar features
   :type use_atom_edge_embedding: bool
   :param use_m_share_rad: Whether all m components within a type-L vector of one channel share radial function weights
   :type use_m_share_rad: bool
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
   :param use_grid_mlp: If `True`, use projecting to grids and performing MLPs for FFN.
   :type use_grid_mlp: bool
   :param use_sep_s2_act: If `True`, use separable S2 activation when `use_gate_act` is False.
   :type use_sep_s2_act: bool
   :param norm_type: Type of normalization layer (['layer_norm', 'layer_norm_sh'])
   :type norm_type: str
   :param alpha_drop: Dropout rate for attention weights
   :type alpha_drop: float
   :param drop_path_rate: Drop path rate
   :type drop_path_rate: float
   :param proj_drop: Dropout rate for outputs of attention and FFN
   :type proj_drop: float

   .. py:method:: forward(x, atomic_numbers, edge_distance, edge_index, batch)



