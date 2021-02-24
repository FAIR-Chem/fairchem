import copy
import math
from typing import Dict, List, Tuple

import dgl
import dgl.function as fn  # for graphs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling
from dgl.nn.pytorch.softmax import edge_softmax
from packaging import version

from . import fibers
from .fibers import Fiber, fiber2head, fiber2tensor, get_fiber_dict
from .from_se3cnn import utils_steerable

### Equivariant basis construction


def get_basis(Y, max_degree):
    """Precompute the SE(3)-equivariant weight basis.

    This is called by get_basis_and_r().

    Args:
        Y: spherical harmonic dict, returned by utils_steerable.precompute_sh()
        max_degree: non-negative int for degree of highest feature type
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
    """
    device = Y[0].device
    # No need to backprop through the basis construction
    with torch.no_grad():
        basis = {}
        for d_in in range(max_degree + 1):
            for d_out in range(max_degree + 1):
                K_Js = []
                for J in range(abs(d_in - d_out), d_in + d_out + 1):
                    # Get spherical harmonic projection matrices
                    Q_J = utils_steerable._basis_transformation_Q_J(
                        J, d_in, d_out
                    )
                    Q_J = Q_J.float().T.to(device)

                    # Create kernel from spherical harmonics
                    K_J = torch.matmul(Y[J], Q_J)
                    K_Js.append(K_J)

                # Reshape so can take linear combinations with a dot product
                size = (
                    -1,
                    1,
                    2 * d_out + 1,
                    1,
                    2 * d_in + 1,
                    2 * min(d_in, d_out) + 1,
                )
                basis[f"{d_in},{d_out}"] = torch.stack(K_Js, -1).view(*size)
        return basis


def get_basis_and_r(G, max_degree):
    """Return equivariant weight basis (basis) and internodal distances (r).

    Call this function *once* at the start of each forward pass of the model.
    It computes the equivariant weight basis, W_J^lk(x), and internodal
    distances, needed to compute varphi_J^lk(x), of eqn 8 of
    https://arxiv.org/pdf/2006.10503.pdf. The return values of this function
    can be shared as input across all SE(3)-Transformer layers in a model.

    Args:
        G: DGL graph instance of type dgl.DGLGraph()
        max_degree: non-negative int for degree of highest feature-type
    Returns:
        dict of equivariant bases, keys are in form '<d_in><d_out>'
        vector of relative distances, ordered according to edge ordering of G
    """
    # Relative positional encodings (vector)
    r_ij = utils_steerable.get_spherical_from_cartesian_torch(G.edata["d"])
    # Spherical harmonic basis
    Y = utils_steerable.precompute_sh(r_ij, 2 * max_degree)
    # Equivariant basis (dict['d_in><d_out>'])
    basis = get_basis(Y, max_degree)
    # Relative distances (scalar)
    r = torch.sqrt(torch.sum(G.edata["d"] ** 2, -1, keepdim=True))
    return basis, r


### SE(3) equivariant operations on graphs in DGL


class GConvSE3(nn.Module):
    """A tensor field network layer as a DGL module.

    GConvSE3 stands for a Graph Convolution SE(3)-equivariant layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
    conv layer in a GCN.

    At each node, the activations are split into different "feature types",
    indexed by the SE(3) representation type: non-negative integers 0, 1, 2, ..
    """

    def __init__(
        self,
        f_in,
        f_out,
        self_interaction: bool = False,
        edge_dim: int = 0,
        nonlin=nn.ReLU(),
    ):
        """SE(3)-equivariant Graph Conv Layer

        Args:
            f_in: list of tuples [(multiplicities, type),...]
            f_out: list of tuples [(multiplicities, type),...]
            self_interaction: include self-interaction in convolution
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.edge_dim = edge_dim
        self.self_interaction = self_interaction

        # Neighbor -> center weights
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f"({di},{do})"] = PairwiseConv(
                    di,
                    mi,
                    do,
                    mo,
                    edge_dim=edge_dim,
                    nonlin=nonlin,
                )

        # Center -> center weights
        self.kernel_self = nn.ParameterDict()
        if self_interaction:
            for m_in, d_in in self.f_in.structure:
                if d_in in self.f_out.degrees:
                    m_out = self.f_out.structure_dict[d_in]
                    W = nn.Parameter(
                        torch.randn(1, m_out, m_in) / np.sqrt(m_in)
                    )
                    self.kernel_self[f"{d_in}"] = W

    def __repr__(self):
        return f"GConvSE3(structure={self.f_out}, self_interaction={self.self_interaction})"

    def udf_u_mul_e(self, d_out):
        """Compute the convolution for a single output feature type.

        This function is set up as a User Defined Function in DGL.

        Args:
            d_out: output feature type
        Returns:
            edge -> node function handle
        """

        def fnc(edges):
            # Neighbor -> center messages
            msg = 0
            for m_in, d_in in self.f_in.structure:
                src = edges.src[f"{d_in}"].view(-1, m_in * (2 * d_in + 1), 1)
                edge = edges.data[f"({d_in},{d_out})"]
                msg = msg + torch.matmul(edge, src)
            msg = msg.view(msg.shape[0], -1, 2 * d_out + 1)

            # Center -> center messages
            if self.self_interaction:
                if f"{d_out}" in self.kernel_self.keys():
                    dst = edges.dst[f"{d_out}"]
                    W = self.kernel_self[f"{d_out}"]
                    msg = msg + torch.matmul(W, dst)

            return {"msg": msg.view(msg.shape[0], -1, 2 * d_out + 1)}

        return fnc

    def forward(self, h, G=None, r=None, basis=None, **kwargs):
        """Forward pass of the linear layer

        Args:
            G: minibatch of (homo)graphs
            h: dict of features
            r: inter-atomic distances
            basis: pre-computed Q * Y
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        with G.local_scope():
            # Add node features to local graph scope
            for k, v in h.items():
                G.ndata[k] = v

            # Add edge features
            if "w" in G.edata.keys():
                w = G.edata["w"]
                feat = torch.cat([w, r], -1)
            else:
                feat = torch.cat(
                    [
                        r,
                    ],
                    -1,
                )

            for (mi, di) in self.f_in.structure:
                for (mo, do) in self.f_out.structure:
                    etype = f"({di},{do})"
                    G.edata[etype] = self.kernel_unary[etype](feat, basis)

            # Perform message-passing for each output feature type
            for d in self.f_out.degrees:
                G.update_all(self.udf_u_mul_e(d), fn.mean("msg", f"out{d}"))

            return {f"{d}": G.ndata[f"out{d}"] for d in self.f_out.degrees}


class RadialFunc(nn.Module):
    """NN parameterized radial profile function."""

    def __init__(
        self, num_freq, in_dim, out_dim, edge_dim: int = 0, nonlin=nn.ReLU()
    ):
        """NN parameterized radial profile function.

        Args:
            num_freq: number of output frequencies
            in_dim: multiplicity of input (num input channels)
            out_dim: multiplicity of output (num output channels)
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        self.num_freq = num_freq
        self.in_dim = in_dim
        self.mid_dim = 32
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.net = nn.Sequential(
            nn.Linear(self.edge_dim + 1, self.mid_dim),
            BN(self.mid_dim),
            nonlin,
            nn.Linear(self.mid_dim, self.mid_dim),
            BN(self.mid_dim),
            nonlin,
            nn.Linear(self.mid_dim, self.num_freq * in_dim * out_dim),
        )

        nn.init.kaiming_uniform_(self.net[0].weight)
        nn.init.kaiming_uniform_(self.net[3].weight)
        nn.init.kaiming_uniform_(self.net[6].weight)

    def __repr__(self):
        return f"RadialFunc(edge_dim={self.edge_dim}, in_dim={self.in_dim}, out_dim={self.out_dim})"

    def forward(self, x):
        y = self.net(x)
        return y.view(-1, self.out_dim, 1, self.in_dim, 1, self.num_freq)


class PairwiseConv(nn.Module):
    """SE(3)-equivariant convolution between two single-type features"""

    def __init__(
        self,
        degree_in: int,
        nc_in: int,
        degree_out: int,
        nc_out: int,
        edge_dim: int = 0,
        nonlin=nn.ReLU(),
    ):
        """SE(3)-equivariant convolution between a pair of feature types.

        This layer performs a convolution from nc_in features of type degree_in
        to nc_out features of type degree_out.

        Args:
            degree_in: degree of input fiber
            nc_in: number of channels on input
            degree_out: degree of out order
            nc_out: number of channels on output
            edge_dim: number of dimensions for edge embedding
        """
        super().__init__()
        # Log settings
        self.degree_in = degree_in
        self.degree_out = degree_out
        self.nc_in = nc_in
        self.nc_out = nc_out

        # Functions of the degree
        self.num_freq = 2 * min(degree_in, degree_out) + 1
        self.d_out = 2 * degree_out + 1
        self.edge_dim = edge_dim

        # Radial profile function
        self.rp = RadialFunc(
            self.num_freq, nc_in, nc_out, self.edge_dim, nonlin
        )

    def forward(self, feat, basis):
        # Get radial weights
        R = self.rp(feat)
        kernel = torch.sum(
            R * basis[f"{self.degree_in},{self.degree_out}"], -1
        )
        return kernel.view(kernel.shape[0], self.d_out * self.nc_out, -1)


class G1x1SE3(nn.Module):
    """Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.

    This is equivalent to a self-interaction layer in TensorField Networks.
    """

    def __init__(self, f_in, f_out, learnable=True):
        """SE(3)-equivariant 1x1 convolution.

        Args:
            f_in: input Fiber() of feature multiplicities and types
            f_out: output Fiber() of feature multiplicities and types
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out

        # Linear mappings: 1 per output feature type
        self.transform = nn.ParameterDict()
        for m_out, d_out in self.f_out.structure:
            m_in = self.f_in.structure_dict[d_out]
            self.transform[str(d_out)] = nn.Parameter(
                torch.randn(m_out, m_in) / np.sqrt(m_in),
                requires_grad=learnable,
            )

    def __repr__(self):
        return f"G1x1SE3(structure={self.f_out})"

    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            if str(k) in self.transform.keys():
                output[k] = torch.matmul(self.transform[str(k)], v)
        return output


class GNormSE3(nn.Module):
    """Graph Norm-based SE(3)-equivariant nonlinearity.

    Nonlinearities are important in SE(3) equivariant GCNs. They are also quite
    expensive to compute, so it is convenient for them to share resources with
    other layers, such as normalization. The general workflow is as follows:

    > for feature type in features:
    >    norm, phase <- feature
    >    output = fnc(norm) * phase

    where fnc: {R+}^m -> R^m is a learnable map from m norms to m scalars.
    """

    def __init__(
        self, fiber, nonlin=nn.ReLU(inplace=True), num_layers: int = 0
    ):
        """Initializer.

        Args:
            fiber: Fiber() of feature multiplicities and types
            nonlin: nonlinearity to use everywhere
            num_layers: non-negative number of linear layers in fnc
        """
        super().__init__()
        self.fiber = fiber
        self.nonlin = nonlin
        self.num_layers = num_layers

        # Regularization for computing phase: gradients explode otherwise
        self.eps = 1e-12

        # Norm mappings: 1 per feature type
        self.transform = nn.ModuleDict()
        for m, d in self.fiber.structure:
            self.transform[str(d)] = self._build_net(int(m))

    def __repr__(self):
        return f"GNormSE3(num_layers={self.num_layers}, nonlin={self.nonlin})"

    def _build_net(self, m: int):
        net = []
        for i in range(self.num_layers):
            net.append(BN(int(m)))
            net.append(self.nonlin)
            # TODO: implement cleaner init
            net.append(nn.Linear(m, m, bias=(i == self.num_layers - 1)))
            nn.init.kaiming_uniform_(net[-1].weight)
        if self.num_layers == 0:
            net.append(BN(int(m)))
            net.append(self.nonlin)
        return nn.Sequential(*net)

    def forward(self, features, **kwargs):
        output = {}
        for k, v in features.items():
            # Compute the norms and normalized features
            # v shape: [...,m , 2*k+1]
            norm = v.norm(2, -1, keepdim=True).clamp_min(self.eps).expand_as(v)
            phase = v / norm

            # Transform on norms
            transformed = self.transform[str(k)](norm[..., 0]).unsqueeze(-1)

            # Nonlinearity on norm
            output[k] = (transformed * phase).view(*v.shape)

        return output


class BN(nn.Module):
    """SE(3)-equvariant batch/layer normalization"""

    def __init__(self, m):
        """SE(3)-equvariant batch/layer normalization

        Args:
            m: int for number of output channels
        """
        super().__init__()
        self.bn = nn.LayerNorm(m)

    def forward(self, x):
        return self.bn(x)


class GConvSE3Partial(nn.Module):
    """Graph SE(3)-equivariant node -> edge layer"""

    def __init__(self, f_in, f_out, edge_dim: int = 0, nonlin=nn.ReLU()):
        """SE(3)-equivariant partial convolution.

        A partial convolution computes the inner product between a kernel and
        each input channel, without summing over the result from each input
        channel. This unfolded structure makes it amenable to be used for
        computing the value-embeddings of the attention mechanism.

        Args:
            f_in: list of tuples [(multiplicities, type),...]
            f_out: list of tuples [(multiplicities, type),...]
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.edge_dim = edge_dim

        # Node -> edge weights
        self.kernel_unary = nn.ModuleDict()
        for (mi, di) in self.f_in.structure:
            for (mo, do) in self.f_out.structure:
                self.kernel_unary[f"({di},{do})"] = PairwiseConv(
                    di,
                    mi,
                    do,
                    mo,
                    edge_dim=edge_dim,
                    nonlin=nonlin,
                )

    def __repr__(self):
        return f"GConvSE3Partial(structure={self.f_out})"

    def udf_u_mul_e(self, d_out):
        """Compute the partial convolution for a single output feature type.

        This function is set up as a User Defined Function in DGL.

        Args:
            d_out: output feature type
        Returns:
            node -> edge function handle
        """

        def fnc(edges):
            # Neighbor -> center messages
            msg = 0
            for m_in, d_in in self.f_in.structure:
                src = edges.src[f"{d_in}"].view(-1, m_in * (2 * d_in + 1), 1)
                edge = edges.data[f"({d_in},{d_out})"]
                msg = msg + torch.matmul(edge, src)
            msg = msg.view(msg.shape[0], -1, 2 * d_out + 1)

            return {f"out{d_out}": msg.view(msg.shape[0], -1, 2 * d_out + 1)}

        return fnc

    def forward(self, h, G=None, r=None, basis=None, **kwargs):
        """Forward pass of the linear layer

        Args:
            h: dict of node-features
            G: minibatch of (homo)graphs
            r: inter-atomic distances
            basis: pre-computed Q * Y
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        with G.local_scope():
            # Add node features to local graph scope
            for k, v in h.items():
                G.ndata[k] = v

            # Add edge features
            if "w" in G.edata.keys():
                w = G.edata["w"]  # shape: [#edges_in_batch, #bond_types]
                feat = torch.cat([w, r], -1)
            else:
                feat = torch.cat(
                    [
                        r,
                    ],
                    -1,
                )
            for (mi, di) in self.f_in.structure:
                for (mo, do) in self.f_out.structure:
                    etype = f"({di},{do})"
                    G.edata[etype] = self.kernel_unary[etype](feat, basis)

            # Perform message-passing for each output feature type
            for d in self.f_out.degrees:
                G.apply_edges(self.udf_u_mul_e(d))

            return {f"{d}": G.edata[f"out{d}"] for d in self.f_out.degrees}


class GMABSE3(nn.Module):
    """An SE(3)-equivariant multi-headed self-attention module for DGL graphs."""

    def __init__(self, f_value: Fiber, f_key: Fiber, n_heads: int):
        """SE(3)-equivariant MAB (multi-headed attention block) layer.

        Args:
            f_value: Fiber() object for value-embeddings
            f_key: Fiber() object for key-embeddings
            n_heads: number of heads
        """
        super().__init__()
        self.f_value = f_value
        self.f_key = f_key
        self.n_heads = n_heads
        self.new_dgl = version.parse(dgl.__version__) > version.parse("0.4.4")

    def __repr__(self):
        return f"GMABSE3(n_heads={self.n_heads}, structure={self.f_value})"

    def udf_u_mul_e(self, d_out):
        """Compute the weighted sum for a single output feature type.

        This function is set up as a User Defined Function in DGL.

        Args:
            d_out: output feature type
        Returns:
            edge -> node function handle
        """

        def fnc(edges):
            # Neighbor -> center messages
            attn = edges.data["a"]
            value = edges.data[f"v{d_out}"]

            # Apply attention weights
            msg = attn.unsqueeze(-1).unsqueeze(-1) * value

            return {"m": msg}

        return fnc

    def forward(self, v, k: Dict = None, q: Dict = None, G=None, **kwargs):
        """Forward pass of the linear layer

        Args:
            G: minibatch of (homo)graphs
            v: dict of value edge-features
            k: dict of key edge-features
            q: dict of query node-features
        Returns:
            tensor with new features [B, n_points, n_features_out]
        """
        with G.local_scope():
            # Add node features to local graph scope
            ## We use the stacked tensor representation for attention
            for m, d in self.f_value.structure:
                G.edata[f"v{d}"] = v[f"{d}"].view(
                    -1, self.n_heads, m // self.n_heads, 2 * d + 1
                )
            G.edata["k"] = fiber2head(
                k, self.n_heads, self.f_key, squeeze=True
            )
            G.ndata["q"] = fiber2head(
                q, self.n_heads, self.f_key, squeeze=True
            )

            # Compute attention weights
            ## Inner product between (key) neighborhood and (query) center
            G.apply_edges(fn.e_dot_v("k", "q", "e"))

            ## Apply softmax
            e = G.edata.pop("e")
            if self.new_dgl:
                # in dgl 5.3, e has an extra dimension compared to dgl 4.3
                # the following, we get rid of this be reshaping
                n_edges = G.edata["k"].shape[0]
                e = e.view([n_edges, self.n_heads])
            e = e / np.sqrt(self.f_key.n_features)
            G.edata["a"] = edge_softmax(G, e)

            # Perform attention-weighted message-passing
            for d in self.f_value.degrees:
                G.update_all(self.udf_u_mul_e(d), fn.sum("m", f"out{d}"))

            output = {}
            for m, d in self.f_value.structure:
                output[f"{d}"] = G.ndata[f"out{d}"].view(-1, m, 2 * d + 1)

            return output


class GSE3Res(nn.Module):
    """Graph attention block with SE(3)-equivariance and skip connection"""

    def __init__(
        self,
        f_in: Fiber,
        f_out: Fiber,
        edge_dim: int = 0,
        div: float = 4,
        n_heads: int = 1,
        learnable_skip=True,
    ):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.div = div
        self.n_heads = n_heads

        # f_mid_out has same structure as 'f_out' but #channels divided by 'div'
        # this will be used for the values
        f_mid_out = {
            k: int(v // div) for k, v in self.f_out.structure_dict.items()
        }
        self.f_mid_out = Fiber(dictionary=f_mid_out)

        # f_mid_in has same structure as f_mid_out, but only degrees which are in f_in
        # this will be used for keys and queries
        # (queries are merely projected, hence degrees have to match input)
        f_mid_in = {
            d: m for d, m in f_mid_out.items() if d in self.f_in.degrees
        }
        self.f_mid_in = Fiber(dictionary=f_mid_in)

        self.edge_dim = edge_dim

        self.GMAB = nn.ModuleDict()

        # Projections
        self.GMAB["v"] = GConvSE3Partial(
            f_in, self.f_mid_out, edge_dim=edge_dim
        )
        self.GMAB["k"] = GConvSE3Partial(
            f_in, self.f_mid_in, edge_dim=edge_dim
        )
        self.GMAB["q"] = G1x1SE3(f_in, self.f_mid_in)

        # Attention
        self.GMAB["attn"] = GMABSE3(
            self.f_mid_out, self.f_mid_in, n_heads=n_heads
        )

        # Skip connections
        self.project = G1x1SE3(self.f_mid_out, f_out, learnable=learnable_skip)
        self.add = GSum(f_out, f_in)
        # the following checks whether the skip connection would change
        # the output fibre strucure; the reason can be that the input has
        # more channels than the ouput (for at least one degree); this would
        # then cause a (hard to debug) error in the next layer
        assert (
            self.add.f_out.structure_dict == f_out.structure_dict
        ), "skip connection would change output structure"

    def forward(self, features, G, **kwargs):
        # Embeddings
        v = self.GMAB["v"](features, G=G, **kwargs)
        k = self.GMAB["k"](features, G=G, **kwargs)
        q = self.GMAB["q"](features, G=G)

        # Attention
        z = self.GMAB["attn"](v, k=k, q=q, G=G)

        # Skip + residual
        z = self.project(z)
        z = self.add(z, features)
        return z


### Helper and wrapper functions


class GSum(nn.Module):
    """SE(3)-equvariant graph residual sum function."""

    def __init__(self, f_x: Fiber, f_y: Fiber):
        """SE(3)-equvariant graph residual sum function.

        Args:
            f_x: Fiber() object for fiber of summands
            f_y: Fiber() object for fiber of summands
        """
        super().__init__()
        self.f_x = f_x
        self.f_y = f_y
        self.f_out = Fiber.combine_max(f_x, f_y)

    def __repr__(self):
        return f"GSum(structure={self.f_out})"

    def forward(self, x, y):
        out = {}
        for k in self.f_out.degrees:
            k = str(k)
            if (k in x) and (k in y):
                if x[k].shape[1] > y[k].shape[1]:
                    diff = x[k].shape[1] - y[k].shape[1]
                    zeros = torch.zeros(x[k].shape[0], diff, x[k].shape[2]).to(
                        y[k].device
                    )
                    y[k] = torch.cat([y[k], zeros], 1)
                elif x[k].shape[1] < y[k].shape[1]:
                    diff = y[k].shape[1] - x[k].shape[1]
                    zeros = torch.zeros(x[k].shape[0], diff, x[k].shape[2]).to(
                        y[k].device
                    )
                    x[k] = torch.cat([x[k], zeros], 1)

                out[k] = x[k] + y[k]
            elif k in x:
                out[k] = x[k]
            elif k in y:
                out[k] = y[k]
        return out


class GAvgPooling(nn.Module):
    """Graph Average Pooling module."""

    def __init__(self, type="0"):
        super().__init__()
        self.pool = AvgPooling()
        self.type = type

    def forward(self, features, G, **kwargs):
        if self.type == "0":
            h = features["0"][..., -1]
            pooled = self.pool(G, h)
        elif self.type == "1":
            pooled = []
            for i in range(3):
                h_i = features["1"][..., i]
                pooled.append(self.pool(G, h_i).unsqueeze(-1))
            pooled = torch.cat(pooled, axis=-1)
            pooled = {"1": pooled}
        else:
            print("GAvgPooling for type > 0 not implemented")
            exit()
        return pooled


class GMaxPooling(nn.Module):
    """Graph Max Pooling module."""

    def __init__(self):
        super().__init__()
        self.pool = MaxPooling()

    def forward(self, features, G, **kwargs):
        h = features["0"][..., -1]
        return self.pool(G, h)
