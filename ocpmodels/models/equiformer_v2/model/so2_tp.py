'''
    Use the parametrization of tensor products for SO(2) convolution.

    TODO: 
        1. Check whether they are the same as e3nn
        2. Use e3nn initialization
'''
import torch
import torch.nn as nn
from e3nn import o3
import math
import copy

from torch.nn import Linear
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding
)
from .radial_function import RadialFunction


def clebsch_gordan_coefficients_array(lmax, normalize=True):
    '''
        Return Clebsch-Gordan coefficients with L up to `lmax`
    '''
    cg_array = torch.zeros((lmax + 1) ** 2, (lmax + 1) ** 2, (lmax + 1) ** 2)
    for l1 in range(lmax + 1):
        start_idx_1 = l1 ** 2
        length_1 = 2 *l1 + 1
        for l2 in range(lmax + 1):
            start_idx_2 = l2 ** 2
            length_2 = 2 * l2 + 1
            for l3 in range(lmax + 1):
                start_idx_3 = l3 ** 2
                length_3 = 2 * l3 + 1

                # selection rule to prevent calling `o3.wigner_3j()` 
                if not (abs(l1 - l2) <= l3):
                    continue
                if not (l3 <= (l1 + l2)):
                    continue

                cg_coeff = o3.wigner_3j(l1, l2, l3)
                if normalize:
                    # This is for alpha in e3nn tensor product: 
                    # https://github.com/e3nn/e3nn/blob/main/e3nn/o3/_tensor_product/_tensor_product.py#L296-L297
                    cg_coeff = cg_coeff * math.sqrt(2 * l3 + 1) 
                    # This is for normalizing spherical harmonics
                    cg_coeff = cg_coeff * math.sqrt(2 * l2 + 1)

                cg_array[
                    start_idx_1 : (start_idx_1 + length_1),
                    start_idx_2 : (start_idx_2 + length_2), 
                    start_idx_3 : (start_idx_3 + length_3), 
                ] = cg_coeff

    cg_array = torch.permute(cg_array, (2, 0, 1)) # (L_o, L_i, L_f)

    return cg_array
                

def extract_m_index(mappingReduced: CoefficientMappingModule, m: int):
    mask = mappingReduced.m_complex.eq(m)
    indices = torch.arange(len(mask))
    mask_indices = torch.masked_select(indices, mask)
    return mask_indices


def index_select_list(tensor, index_list):
    for i in range(len(index_list)):
        if i == 0:
            output = torch.index_select(tensor, i, index_list[i])
        else:
            output = torch.index_select(output, i, index_list[i])
    return output


def rescale_tensor_product_weights(tp_weight_data, lmax):
    '''
        `tp_weight` is of shape (L_o, C_o, L_i, C_i, L_f, C_f)
    '''
    irreps = o3.Irreps.spherical_harmonics(lmax, p=1)
    fctp = o3.FullyConnectedTensorProduct(irreps, irreps, irreps, 
        irrep_normalization='none',    # we consider this in `clebsch_gordan_coefficients_array`
        path_normalization='element'
    )
    for instruction in fctp.instructions:
        tp_weight_data[
            instruction.i_out, :, 
            instruction.i_in1, :, 
            instruction.i_in2, :
        ] *= instruction.path_weight
    return tp_weight_data


class SO2_m_Convolution(torch.nn.Module):
    """
    1. SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m
    2. Use tensor product reparametrization and store Clebsch-Gordan coefficients
    3. m should be >= 0

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
    """
    def __init__(
        self,
        m, 
        sphere_channels,
        m_output_channels,
        lmax_list, 
        mmax_list
    ):
        super(SO2_m_Convolution, self).__init__()
        
        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        assert self.num_resolutions == 1, 'This implementation only supports single resolutions.'
        assert self.m >= 0
        assert self.m <= self.mmax_list[0]
        assert self.m <= self.lmax_list[0]

        self.lmax = self.lmax_list[0]
        self.num_coefficents = self.lmax - self.m + 1
        self.num_in_channels = self.num_coefficents * self.sphere_channels # not used since we feed the weights during forward()

        # Clebsch-Gordan coefficients
        cg_array = clebsch_gordan_coefficients_array(self.lmax)
        mappingReduced = CoefficientMappingModule(lmax_list=[self.lmax], mmax_list=[self.lmax])
        #   For m >= 0
        m_pos_index = extract_m_index(mappingReduced, self.m) # for L_i and L_o
        m_0_index   = extract_m_index(mappingReduced, 0)      # for L_f
        cg_array_m_pos = index_select_list(cg_array, [m_pos_index, m_pos_index, m_0_index]) # (L_o, L_i, L_f)
        cg_array_m_pos = cg_array_m_pos.view(
            cg_array_m_pos.shape[0], 1,
            cg_array_m_pos.shape[1], 1,  
            cg_array_m_pos.shape[2], 1
        )
        self.register_buffer('cg_array_m_pos', cg_array_m_pos)
        #   For m < 0
        if self.m > 0:
            m_neg_index = extract_m_index(mappingReduced, -self.m) # for L_i and L_o
            cg_array_m_neg = index_select_list(cg_array, [m_neg_index, m_pos_index, m_0_index]) # (L_o, L_i, L_f)
            cg_array_m_neg = cg_array_m_neg.view(
                cg_array_m_neg.shape[0], 1,
                cg_array_m_neg.shape[1], 1,  
                cg_array_m_neg.shape[2], 1
            )
            self.register_buffer('cg_array_m_neg', cg_array_m_neg)
        else:
            self.cg_array_m_neg = None

        # extract a subset of shared weights (only a subset of L contributes to m)
        self.weight_start_idx = self.m
        self.weight_length = (self.lmax + 1 - self.m)

    
    def forward(self, x_m, tp_weight, tp_bias=None):
        '''
            1. Use `tp_weight` to generate weights for SO(2) convolution
            2. `tp_bias` is for m = 0
            3. The shape of `x_m` is (N, 2, L * C) for m > 0
            4. The shape of `x_m` is (N, 1, L * C) for m = 0 
        '''
        # shape: (L_o, C_o, L_i, C_i, L_f, C_f)
        sub_tp_weight = tp_weight[
            self.weight_start_idx:(self.weight_start_idx + self.weight_length), :, 
            self.weight_start_idx:(self.weight_start_idx + self.weight_length), :,
            :, :
        ]
        assert sub_tp_weight.shape[1] == self.m_output_channels
        assert sub_tp_weight.shape[3] == self.sphere_channels
        
        # For m >= 0:
        cg_array_m_pos = self.cg_array_m_pos
        #   sum over L_f
        sub_tp_weight_pos = torch.einsum('ijklmn, ijklmn -> ijkln', sub_tp_weight, cg_array_m_pos)
        sub_tp_weight_pos = sub_tp_weight_pos.reshape(
            sub_tp_weight_pos.shape[0] * sub_tp_weight_pos.shape[1],
            sub_tp_weight_pos.shape[2] * sub_tp_weight_pos.shape[3]
        )    # (L_o * C_o, L_i * C_i)
        
        # For m < 0:
        if self.m > 0:
            cg_array_m_neg = self.cg_array_m_neg
            #   sum over L_f
            sub_tp_weight_neg = torch.einsum('ijklmn, ijklmn -> ijkln', sub_tp_weight, cg_array_m_neg)
            sub_tp_weight_neg = sub_tp_weight_neg.reshape(
                sub_tp_weight_neg.shape[0] * sub_tp_weight_neg.shape[1],
                sub_tp_weight_neg.shape[2] * sub_tp_weight_neg.shape[3]
            )    # (L_o * C_o, L_i * C_i)
            sub_tp_weight = torch.cat((sub_tp_weight_pos, sub_tp_weight_neg), dim=0)
            assert tp_bias is None
        else:
            sub_tp_weight = sub_tp_weight_pos
            zero_tensor = torch.zeros((sub_tp_weight.shape[0] - self.m_output_channels), device=tp_bias.device)
            tp_bias = torch.cat((tp_bias, zero_tensor), dim=0)
        
        y_m = torch.nn.functional.linear(x_m, sub_tp_weight, tp_bias)
        
        if self.m > 0:
            num_out_channels = sub_tp_weight.shape[0]
            y_r = y_m.narrow(2, 0, num_out_channels // 2)
            y_i = y_m.narrow(2, num_out_channels // 2, num_out_channels // 2)
            y_m_r = y_r.narrow(1, 0, 1) - y_i.narrow(1, 1, 1)
            y_m_i = y_r.narrow(1, 1, 1) + y_i.narrow(1, 0, 1)
            y_m = torch.cat((y_m_r, y_m_i), dim=1)
        
        return y_m
    

    def __repr__(self):
        return f"{self.__class__.__name__}(m={self.m}, in_features={self.weight_length}x{self.sphere_channels}, out_features={self.weight_length}x{self.m_output_channels})"


class SO2_Convolution_TensorProduct(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders) with tensor product reparametrization

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        mappingReduced (CoefficientMappingModule): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels (int): If not None, return `out_embedding` (SO3_Embedding) and `extra_m0_features` (Tensor).
    """
    def __init__(
        self,
        sphere_channels,
        m_output_channels,
        lmax_list,
        mmax_list,
        mappingReduced,
        internal_weights=True,
        edge_channels_list=None,
        extra_m0_output_channels=None
    ):
        super(SO2_Convolution_TensorProduct, self).__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.mappingReduced = mappingReduced
        self.num_resolutions = len(lmax_list)
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.extra_m0_output_channels = extra_m0_output_channels
        assert self.num_resolutions == 1, 'This implementation only supports single resolutions.'
        assert self.internal_weights, 'This implementation does not support radial functions.'

        self.lmax = self.lmax_list[0]
        # Tensor product
        self.tp_weight = torch.nn.Parameter(torch.randn(
            (self.lmax + 1), self.m_output_channels,    # for L_o
            (self.lmax + 1), self.sphere_channels,      # for L_i
            (self.lmax + 1), 1                          # for L_f
        ))
        self.tp_weight.data.uniform_(-1.0, 1.0)
        self.tp_weight.data.mul_(1.0 / math.sqrt(self.sphere_channels))
        rescale_tensor_product_weights(self.tp_weight.data, self.lmax)
        self.tp_bias = torch.nn.Parameter(torch.zeros(self.m_output_channels))  # only for L = 0

        # Tensor product for extra m = 0 (or scalars)
        # this can be simply reduced to Linear operating on only m = 0
        if self.extra_m0_output_channels is not None:
            self.extra_m0_linear = torch.nn.Linear(
                (self.lmax + 1) * self.sphere_channels, 
                self.extra_m0_output_channels
            )

        # SO(2) convolution with tensor product reparametrization
        self.so2_m_conv = torch.nn.ModuleList()
        for m in range(0, max(self.mmax_list) + 1):
            self.so2_m_conv.append(
                SO2_m_Convolution(
                    m, 
                    self.sphere_channels,
                    self.m_output_channels,
                    self.lmax_list, 
                    self.mmax_list
                )
            )


    def forward(self, x, x_edge):
        '''
            `x_edge` is not used but is included for consistency with previous implementation.
        '''
        num_edges = len(x.embedding)
        out = []

        # Reshape the spherical harmonics based on m (order)
        x._m_primary(self.mappingReduced)

        # SO(2) convolution
        offset = 0
        for m in range(max(self.mmax_list) + 1):
            
            num_m_components = 1 if m == 0 else 2
            x_m = x.embedding.narrow(1, offset, self.mappingReduced.m_size[m] * num_m_components)
            x_m = x_m.reshape(num_edges, num_m_components, -1)
            tp_bias = self.tp_bias if m == 0 else None
            y_m = self.so2_m_conv[m](x_m, self.tp_weight, tp_bias)
            y_m = y_m.view(num_edges, -1, self.m_output_channels)

            if (m == 0) and (self.extra_m0_output_channels is not None):
                x_0_extra = self.extra_m0_linear(x_m)
                x_0_extra = x_0_extra.view(num_edges, -1)

            out.append(y_m)
            offset = offset + self.mappingReduced.m_size[m] * num_m_components

        out = torch.cat(out, dim=1)
        out_embedding = SO3_Embedding(
            0, 
            x.lmax_list.copy(), 
            self.m_output_channels, 
            device=x.device, 
            dtype=x.dtype
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        out_embedding._l_primary(self.mappingReduced)

        if self.extra_m0_output_channels is not None:
            return out_embedding, x_0_extra
        else:
            return out_embedding
