import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class FeedForward(nn.Module):
    def __init__(
            self,
            embedding_dim,
            ffn_embedding_dim,
            q_noise,
            qn_block_size,
            activation_fn,
            activation_dropout,
            dropout,
            module_name
    ):
        super().__init__()
        self.fc1 = quant_noise(nn.Linear(embedding_dim, ffn_embedding_dim), q_noise, qn_block_size)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=module_name
        )
        self.fc2 = quant_noise(nn.Linear(ffn_embedding_dim, embedding_dim), q_noise, qn_block_size)
        self.dropout_module = FairseqDropout(
            dropout, module_name=module_name
        )

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
