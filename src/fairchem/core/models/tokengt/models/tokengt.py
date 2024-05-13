"""
Modified from https://github.com/microsoft/Graphormer
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)
from fairseq.utils import safe_hasattr

from ..modules import init_graphormer_params, TokenGTGraphEncoder

logger = logging.getLogger(__name__)


@register_model("tokengt")
class TokenGTModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)
        self.encoder_embed_dim = args.encoder_embed_dim

        if args.performer_finetune:
            self.encoder.performer_finetune_setup()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--dropout", type=float, metavar="D",  help="dropout prob")
        parser.add_argument("--attention-dropout", type=float, metavar="D", help="dropout prob for attention weights")
        parser.add_argument("--act-dropout", type=float, metavar="D", help="dropout prob after activation in FFN")

        parser.add_argument("--encoder-ffn-embed-dim", type=int, metavar="N", help="encoder embedding dim for FFN")
        parser.add_argument("--encoder-layers", type=int, metavar="N", help="num encoder layers")
        parser.add_argument("--encoder-attention-heads", type=int, metavar="N", help="num encoder attention heads")
        parser.add_argument("--encoder-embed-dim", type=int, metavar="N", help="encoder embedding dimension")
        parser.add_argument("--share-encoder-input-output-embed", action="store_true",
                            help="share encoder input and output embeddings")

        parser.add_argument("--rand-node-id", action="store_true", help="use random feature node identifiers")
        parser.add_argument("--rand-node-id-dim", type=int, metavar="N", help="dim of random node identifiers")
        parser.add_argument("--orf-node-id", action="store_true", help="use orthogonal random feature node identifiers")
        parser.add_argument("--orf-node-id-dim", type=int, metavar="N", help="dim of orthogonal random node identifier")
        parser.add_argument("--lap-node-id", action="store_true", help="use Laplacian eigenvector node identifiers")
        parser.add_argument("--lap-node-id-k", type=int, metavar="N",
                            help="number of Laplacian eigenvectors to use, from smallest eigenvalues")
        parser.add_argument("--lap-node-id-sign-flip", action="store_true", help="randomly flip the signs of eigvecs")
        parser.add_argument("--lap-node-id-eig-dropout", type=float, metavar="D", help="dropout prob for Lap eigvecs")
        parser.add_argument("--type-id", action="store_true", help="use type identifiers")

        parser.add_argument("--stochastic-depth", action="store_true", help="use stochastic depth regularizer")

        parser.add_argument("--performer", action="store_true", help="linearized self-attention with Performer kernel")
        parser.add_argument("--performer-nb-features", type=int, metavar="N",
                            help="number of random features for Performer, defaults to (d*log(d)) where d is head dim")
        parser.add_argument("--performer-feature-redraw-interval", type=int, metavar="N",
                            help="how frequently to redraw the projection matrix for Performer")
        parser.add_argument("--performer-generalized-attention", action="store_true",
                            help="defaults to softmax approximation, but can be set to True for generalized attention")
        parser.add_argument("--performer-finetune", action="store_true",
                            help="load softmax checkpoint and fine-tune with performer")

        parser.add_argument("--apply-graphormer-init", action="store_true", help="use Graphormer initialization")
        parser.add_argument("--activation-fn", choices=utils.get_available_activation_fns(), help="activation to use")
        parser.add_argument("--encoder-normalize-before", action="store_true", help="apply layernorm before encoder")
        parser.add_argument("--prenorm", action="store_true", help="apply layernorm before self-attention and ffn")
        parser.add_argument("--postnorm", action="store_true", help="apply layernorm after self-attention and ffn")
        parser.add_argument("--return-attention", action="store_true", help="obtain attention maps from all layers",)

    def max_nodes(self):
        return self.encoder.max_nodes

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        logger.info(args)

        encoder = TokenGTEncoder(args)
        return cls(args, encoder)

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class TokenGTEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        assert not (args.prenorm and args.postnorm)
        assert args.prenorm or args.postnorm
        self.max_nodes = args.max_nodes
        self.encoder_layers = args.encoder_layers
        self.num_attention_heads = args.encoder_attention_heads
        self.return_attention = args.return_attention

        if args.prenorm:
            layernorm_style = "prenorm"
        elif args.postnorm:
            layernorm_style = "postnorm"
        else:
            raise NotImplementedError

        self.graph_encoder = TokenGTGraphEncoder(
            # <
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            # < for tokenization
            rand_node_id=args.rand_node_id,
            rand_node_id_dim=args.rand_node_id_dim,
            orf_node_id=args.orf_node_id,
            orf_node_id_dim=args.orf_node_id_dim,
            lap_node_id=args.lap_node_id,
            lap_node_id_k=args.lap_node_id_k,
            lap_node_id_sign_flip=args.lap_node_id_sign_flip,
            lap_node_id_eig_dropout=args.lap_node_id_eig_dropout,
            type_id=args.type_id,
            # >
            # <
            stochastic_depth=args.stochastic_depth,
            performer=args.performer,
            performer_finetune=args.performer_finetune,
            performer_nb_features=args.performer_nb_features,
            performer_feature_redraw_interval=args.performer_feature_redraw_interval,
            performer_generalized_attention=args.performer_generalized_attention,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            layernorm_style=layernorm_style,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
            return_attention=args.return_attention
            # >
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)
        self.masked_lm_pooler = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.lm_head_transform_weight = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(args.encoder_embed_dim, args.num_classes, bias=False)
            else:
                raise NotImplementedError

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep, attn_dict = self.graph_encoder(batched_data, perturb=perturb)

        x = inner_states[-1].transpose(0, 1)  # B x T x C

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
                self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        if self.return_attention:
            return x[:, 0, :], attn_dict
        else:
            return x[:, 0, :]

    def performer_finetune_setup(self):
        self.graph_encoder.performer_finetune_setup()

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict


@register_model_architecture("tokengt", "tokengt")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(args, "share_encoder_input_output_embed", False)
    args.prenorm = getattr(args, "prenorm", False)
    args.postnorm = getattr(args, "postnorm", False)

    args.rand_node_id = getattr(args, "rand_node_id", False)
    args.rand_node_id_dim = getattr(args, "rand_node_id_dim", 64)
    args.orf_node_id = getattr(args, "orf_node_id", False)
    args.orf_node_id_dim = getattr(args, "orf_node_id_dim", 64)
    args.lap_node_id = getattr(args, "lap_node_id", False)
    args.lap_node_id_k = getattr(args, "lap_node_id_k", 8)
    args.lap_node_id_sign_flip = getattr(args, "lap_node_id_sign_flip", False)
    args.lap_node_id_eig_dropout = getattr(args, "lap_node_id_eig_dropout", 0.0)
    args.type_id = getattr(args, "type_id", True)

    args.stochastic_depth = getattr(args, "stochastic_depth", False)

    args.performer = getattr(args, "performer", False)
    args.performer_finetune = getattr(args, "performer_finetune", False)
    args.performer_nb_features = getattr(args, "performer_nb_features", None)
    args.performer_feature_redraw_interval = getattr(args, "performer_feature_redraw_interval", 1000)
    args.performer_generalized_attention = getattr(args, "performer_generalized_attention", False)

    args.return_attention = getattr(args, "return_attention", False)
