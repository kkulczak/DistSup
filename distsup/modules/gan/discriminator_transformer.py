import math

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# from src.utils import LReluCustom
from distsup.modules.gan import transformer_encoder
from distsup.modules.gan.data_types import GanConfig


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(
            self.pe[:, :seq_len],
            requires_grad=False
        ).cuda()
        return x


class TransformerDiscriminator(nn.Module):
    def __init__(
        self,
        gan_config: dict,
    ):
        super(TransformerDiscriminator, self).__init__()
        self.gan_config = GanConfig(**gan_config)
        self.conv_n_feature_1 = self.gan_config.dis_emb_size
        self.conv_n_feature_2 = self.gan_config.dis_hidden_1_size * 4

        self.embeddings = nn.Embedding(
            num_embeddings=self.gan_config.dictionary_size,
            embedding_dim=self.gan_config.dis_emb_size,
        )


        self.positional_embeddings = PositionalEncoder(
            d_model=self.gan_config.dis_emb_size,
            max_seq_len=self.gan_config.max_sentence_length,
        )
        ################################################################
        # Weights initialization with xavier values in embedding matrix
        ################################################################
        nn.init.xavier_uniform_(self.embeddings.weight)
        if self.embeddings.padding_idx is not None:
            with torch.no_grad():
                self.embeddings.weight[
                    self.embeddings.padding_idx
                ].fill_(0)
        ################################################################
        ################################################################

        self.transformer = transformer_encoder.TransformerEncoder(
            encoder_layer=transformer_encoder.TransformerEncoderLayer(
                self.gan_config.dis_emb_size,
                nhead=8,
                dim_feedforward=1024

            ),
            num_layers=3
        )

        self.dense_input_size = (
            self.gan_config.dis_emb_size
        )
        self.dense = nn.Linear(self.dense_input_size, 1)

    def forward(self, x):
        batch_size, phrase_length, element_size = x.shape
        x = torch.matmul(x, self.embeddings.weight)

        pos_enc = self.positional_embeddings(x)

        transformer_out = self.transformer(pos_enc)

        mean = transformer_out.mean(dim=1)

        dense_input = mean.view(
            batch_size, self.dense_input_size
        )

        pred = self.dense(dense_input)

        return pred
