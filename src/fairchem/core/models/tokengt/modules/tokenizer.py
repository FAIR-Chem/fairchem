import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .orf import gaussian_orthogonal_random_matrix_batched


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphFeatureTokenizer(nn.Module):
    """
    Compute node and edge features for each node and edge in the graph.
    """

    def __init__(
            self,
            num_atoms,
            num_edges,
            rand_node_id,
            rand_node_id_dim,
            orf_node_id,
            orf_node_id_dim,
            lap_node_id,
            lap_node_id_k,
            lap_node_id_sign_flip,
            lap_node_id_eig_dropout,
            type_id,
            hidden_dim,
            n_layers
    ):
        super(GraphFeatureTokenizer, self).__init__()

        self.encoder_embed_dim = hidden_dim

        self.atom_encoder = nn.Embedding(num_atoms, hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(num_edges, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.null_token = nn.Embedding(1, hidden_dim)  # this is optional

        self.rand_node_id = rand_node_id
        self.rand_node_id_dim = rand_node_id_dim
        self.orf_node_id = orf_node_id
        self.orf_node_id_dim = orf_node_id_dim
        self.lap_node_id = lap_node_id
        self.lap_node_id_k = lap_node_id_k
        self.lap_node_id_sign_flip = lap_node_id_sign_flip

        self.type_id = type_id

        if self.rand_node_id:
            self.rand_encoder = nn.Linear(2 * rand_node_id_dim, hidden_dim, bias=False)

        if self.lap_node_id:
            self.lap_encoder = nn.Linear(2 * lap_node_id_k, hidden_dim, bias=False)
            self.lap_eig_dropout = nn.Dropout2d(p=lap_node_id_eig_dropout) if lap_node_id_eig_dropout > 0 else None

        if self.orf_node_id:
            self.orf_encoder = nn.Linear(2 * orf_node_id_dim, hidden_dim, bias=False)

        if self.type_id:
            self.order_encoder = nn.Embedding(2, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    @staticmethod
    def get_batch(node_feature, edge_index, edge_feature, node_num, edge_num, perturb=None):
        """
        :param node_feature: Tensor([sum(node_num), D])
        :param edge_index: LongTensor([2, sum(edge_num)])
        :param edge_feature: Tensor([sum(edge_num), D])
        :param node_num: list
        :param edge_num: list
        :param perturb: Tensor([B, max(node_num), D])
        :return: padded_index: LongTensor([B, T, 2]), padded_feature: Tensor([B, T, D]), padding_mask: BoolTensor([B, T])
        """
        seq_len = [n + e for n, e in zip(node_num, edge_num)]
        b = len(seq_len)
        d = node_feature.size(-1)
        max_len = max(seq_len)
        max_n = max(node_num)
        device = edge_index.device

        token_pos = torch.arange(max_len, device=device)[None, :].expand(b, max_len)  # [B, T]

        seq_len = torch.tensor(seq_len, device=device, dtype=torch.long)[:, None]  # [B, 1]
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[:, None]  # [B, 1]
        edge_num = torch.tensor(edge_num, device=device, dtype=torch.long)[:, None]  # [B, 1]

        node_index = torch.arange(max_n, device=device, dtype=torch.long)[None, :].expand(b, max_n)  # [B, max_n]
        node_index = node_index[None, node_index < node_num].repeat(2, 1)  # [2, sum(node_num)]

        padded_node_mask = torch.less(token_pos, node_num)
        padded_edge_mask = torch.logical_and(
            torch.greater_equal(token_pos, node_num),
            torch.less(token_pos, node_num + edge_num)
        )

        padded_index = torch.zeros(b, max_len, 2, device=device, dtype=torch.long)  # [B, T, 2]
        padded_index[padded_node_mask, :] = node_index.t()
        padded_index[padded_edge_mask, :] = edge_index.t()

        if perturb is not None:
            perturb_mask = padded_node_mask[:, :max_n]  # [B, max_n]
            node_feature = node_feature + perturb[perturb_mask].type(node_feature.dtype)  # [sum(node_num), D]

        padded_feature = torch.zeros(b, max_len, d, device=device, dtype=node_feature.dtype)  # [B, T, D]
        padded_feature[padded_node_mask, :] = node_feature
        padded_feature[padded_edge_mask, :] = edge_feature

        padding_mask = torch.greater_equal(token_pos, seq_len)  # [B, T]
        return padded_index, padded_feature, padding_mask, padded_node_mask, padded_edge_mask

    @staticmethod
    @torch.no_grad()
    def get_node_mask(node_num, device):
        b = len(node_num)
        max_n = max(node_num)
        node_index = torch.arange(max_n, device=device, dtype=torch.long)[None, :].expand(b, max_n)  # [B, max_n]
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[:, None]  # [B, 1]
        node_mask = torch.less(node_index, node_num)  # [B, max_n]
        return node_mask

    @staticmethod
    @torch.no_grad()
    def get_random_sign_flip(eigvec, node_mask):
        b, max_n = node_mask.size()
        d = eigvec.size(1)

        sign_flip = torch.rand(b, d, device=eigvec.device, dtype=eigvec.dtype)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        sign_flip = sign_flip[:, None, :].expand(b, max_n, d)
        sign_flip = sign_flip[node_mask]
        return sign_flip

    def handle_eigvec(self, eigvec, node_mask, sign_flip):
        if sign_flip and self.training:
            sign_flip = self.get_random_sign_flip(eigvec, node_mask)
            eigvec = eigvec * sign_flip
        else:
            pass
        return eigvec

    @staticmethod
    @torch.no_grad()
    def get_orf_batched(node_mask, dim, device, dtype):
        b, max_n = node_mask.size(0), node_mask.size(1)
        orf = gaussian_orthogonal_random_matrix_batched(b, dim, dim, device=device, dtype=dtype)  # [B, D, D]
        orf = orf[:, None, ...].expand(b, max_n, dim, dim)  # [B, max(n_node), D, D]
        orf = orf[node_mask]  # [sum(n_node), D, D]
        return orf

    @staticmethod
    def get_index_embed(node_id, node_mask, padded_index):
        """
        :param node_id: Tensor([sum(node_num), D])
        :param node_mask: BoolTensor([B, max_n])
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, 2D])
        """
        b, max_n = node_mask.size()
        max_len = padded_index.size(1)
        d = node_id.size(-1)

        padded_node_id = torch.zeros(b, max_n, d, device=node_id.device, dtype=node_id.dtype)  # [B, max_n, D]
        padded_node_id[node_mask] = node_id

        padded_node_id = padded_node_id[:, :, None, :].expand(b, max_n, 2, d)
        padded_index = padded_index[..., None].expand(b, max_len, 2, d)
        index_embed = padded_node_id.gather(1, padded_index)  # [B, T, 2, D]
        index_embed = index_embed.view(b, max_len, 2 * d)
        return index_embed

    def get_type_embed(self, padded_index):
        """
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, D])
        """
        order = torch.eq(padded_index[..., 0], padded_index[..., 1]).long()  # [B, T]
        order_embed = self.order_encoder(order)
        return order_embed

    def add_special_tokens(self, padded_feature, padding_mask):
        """
        :param padded_feature: Tensor([B, T, D])
        :param padding_mask: BoolTensor([B, T])
        :return: padded_feature: Tensor([B, 2/3 + T, D]), padding_mask: BoolTensor([B, 2/3 + T])
        """
        b, _, d = padded_feature.size()

        num_special_tokens = 2
        graph_token_feature = self.graph_token.weight.expand(b, 1, d)  # [1, D]
        null_token_feature = self.null_token.weight.expand(b, 1, d)  # [1, D], this is optional
        special_token_feature = torch.cat((graph_token_feature, null_token_feature), dim=1)  # [B, 2, D]
        special_token_mask = torch.zeros(b, num_special_tokens, dtype=torch.bool, device=padded_feature.device)

        padded_feature = torch.cat((special_token_feature, padded_feature), dim=1)  # [B, 2 + T, D]
        padding_mask = torch.cat((special_token_mask, padding_mask), dim=1)  # [B, 2 + T]
        return padded_feature, padding_mask

    def forward(self, batched_data, perturb=None):
        (
            node_data,
            in_degree,
            out_degree,
            node_num,
            lap_eigvec,
            lap_eigval,
            edge_index,
            edge_data,
            edge_num
        ) = (
            batched_data["node_data"],
            batched_data["in_degree"],
            batched_data["out_degree"],
            batched_data["node_num"],
            batched_data["lap_eigvec"],
            batched_data["lap_eigval"],
            batched_data["edge_index"],
            batched_data["edge_data"],
            batched_data["edge_num"]
        )

        node_feature = self.atom_encoder(node_data).sum(-2)  # [sum(n_node), D]
        edge_feature = self.edge_encoder(edge_data).sum(-2)  # [sum(n_edge), D]
        device = node_feature.device
        dtype = node_feature.dtype

        padded_index, padded_feature, padding_mask, _, _ = self.get_batch(
            node_feature, edge_index, edge_feature, node_num, edge_num, perturb
        )
        node_mask = self.get_node_mask(node_num, node_feature.device)  # [B, max(n_node)]

        if self.rand_node_id:
            rand_node_id = torch.rand(sum(node_num), self.rand_node_id_dim, device=device, dtype=dtype)  # [sum(n_node), D]
            rand_node_id = F.normalize(rand_node_id, p=2, dim=1)
            rand_index_embed = self.get_index_embed(rand_node_id, node_mask, padded_index)  # [B, T, 2D]
            padded_feature = padded_feature + self.rand_encoder(rand_index_embed)

        if self.orf_node_id:
            b, max_n = len(node_num), max(node_num)
            orf = gaussian_orthogonal_random_matrix_batched(
                b, max_n, max_n, device=device, dtype=dtype
            )  # [b, max(n_node), max(n_node)]
            orf_node_id = orf[node_mask]  # [sum(n_node), max(n_node)]
            if self.orf_node_id_dim > max_n:
                orf_node_id = F.pad(orf_node_id, (0, self.orf_node_id_dim - max_n), value=float('0'))  # [sum(n_node), Do]
            else:
                orf_node_id = orf_node_id[..., :self.orf_node_id_dim]  # [sum(n_node), Do]
            orf_node_id = F.normalize(orf_node_id, p=2, dim=1)
            orf_index_embed = self.get_index_embed(orf_node_id, node_mask, padded_index)  # [B, T, 2Do]
            padded_feature = padded_feature + self.orf_encoder(orf_index_embed)

        if self.lap_node_id:
            lap_dim = lap_eigvec.size(-1)
            if self.lap_node_id_k > lap_dim:
                eigvec = F.pad(lap_eigvec, (0, self.lap_node_id_k - lap_dim), value=float('0'))  # [sum(n_node), Dl]
            else:
                eigvec = lap_eigvec[:, :self.lap_node_id_k]  # [sum(n_node), Dl]
            if self.lap_eig_dropout is not None:
                eigvec = self.lap_eig_dropout(eigvec[..., None, None]).view(eigvec.size())
            lap_node_id = self.handle_eigvec(eigvec, node_mask, self.lap_node_id_sign_flip)
            lap_index_embed = self.get_index_embed(lap_node_id, node_mask, padded_index)  # [B, T, 2Dl]
            padded_feature = padded_feature + self.lap_encoder(lap_index_embed)

        if self.type_id:
            padded_feature = padded_feature + self.get_type_embed(padded_index)

        padded_feature, padding_mask = self.add_special_tokens(padded_feature, padding_mask)  # [B, 2+T, D], [B, 2+T]

        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float('0'))
        return padded_feature, padding_mask, padded_index  # [B, 2+T, D], [B, 2+T], [B, T, 2]
