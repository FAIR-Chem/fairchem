import torch

EPS = 1e-15

# TODO: build a class -- use it directly inside core model function. This way we can tweak it as much as desired
# dense_hoscpool will be a fct of this class


def dense_hoscpool(
    x, adj, s, mu=0.1, alpha=0.5, new_ortho=False, mask=None, sn=None, device=None
):
    r"""Mixed higher order spectral clustering pooling operator
    Combines triangle and edge cut
    Based on dense learned cluster assignments
    Returns pooled node feature matrix, coarsened symmetrically normalized
    adjacency matrix and three auxiliary objectives: (1) The edge mincut loss
    (2) The triangle mincut loss (3) The orthogonality terms
    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Symmetrically normalized adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mu (Tensor): scalar that controls the importance given to regularization loss
        alpha (Tensor): scalar in [0,1] controlling the importance granted
            to higher-order information
        new_ortho (BoolTensor): either to use new proposed loss or old one
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    if sn:
        torch.where(adj[0] != 0, torch.ones(1).to(device), torch.zeros(1).to(device))

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    # Output adjacency and feature matrices
    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Motif adj matrix - not sym. normalised
    motif_adj = torch.mul(torch.matmul(adj, adj), adj)
    motif_out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), motif_adj), s)

    mincut_loss = ho_mincut_loss = 0
    # 1st order MinCUT loss
    if alpha < 1:
        diag_SAS = torch.einsum("ijj->ij", out_adj.clone())
        d_flat = torch.einsum("ijk->ij", adj.clone())
        d = _rank3_diag(d_flat)
        sds = torch.matmul(torch.matmul(s.transpose(1, 2), d), s)
        diag_SDS = torch.einsum("ijk->ij", sds) + EPS
        mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        mincut_loss = 1 / k * torch.mean(mincut_loss)

    # Higher order cut
    if alpha > 0:
        diag_SAS = torch.einsum("ijj->ij", motif_out_adj)
        d_flat = torch.einsum("ijk->ij", motif_adj)
        d = _rank3_diag(d_flat)
        diag_SDS = (
            torch.einsum("ijk->ij", torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
            + EPS
        )
        ho_mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        ho_mincut_loss = 1 / k * torch.mean(ho_mincut_loss)

    # Combine ho and fo mincut loss.
    # We do not learn these coefficients yet
    hosc_loss = alpha * mincut_loss + (1 - alpha) * ho_mincut_loss

    # Orthogonality loss
    if mu != 0:
        if new_ortho:
            if s.shape[0] == 1:
                ortho_loss = (
                    (-torch.sum(torch.norm(s, p="fro", dim=-2)) / (num_nodes**0.5))
                    + k**0.5
                ) / (
                    k**0.5 - 1
                )  # [0,1]
            elif mask != None:
                ortho_loss = sum(
                    [
                        (
                            (
                                -torch.sum(
                                    torch.norm(
                                        s[i][: mask[i].nonzero().shape[0]],
                                        p="fro",
                                        dim=-2,
                                    )
                                )
                                / (mask[i].nonzero().shape[0] ** 0.5)
                                + k**0.5
                            )
                            / (k**0.5 - 1)
                        )
                        for i in range(batch_size)
                    ]
                ) / float(batch_size)
            else:
                ortho_loss = sum(
                    [
                        (
                            (
                                -torch.sum(torch.norm(s[i], p="fro", dim=-2))
                                / (num_nodes**0.5)
                                + k**0.5
                            )
                            / (k**0.5 - 1)
                        )
                        for i in range(batch_size)
                    ]
                ) / float(batch_size)
        else:
            # Orthogonality regularization.
            ss = torch.matmul(s.transpose(1, 2), s)
            i_s = torch.eye(k).type_as(ss)
            ortho_loss = torch.norm(
                ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
                dim=(-1, -2),
            )
            ortho_loss = torch.mean(ortho_loss)
    else:
        ortho_loss = torch.tensor(0)

    # Fix and normalize coarsened adjacency matrix. - do not normalize
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0

    # if not new_mincut or sn:
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d + EPS)[:, None]
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, hosc_loss, mu * ortho_loss, s


def _rank3_trace(x):
    return torch.einsum("ijj->i", x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out
