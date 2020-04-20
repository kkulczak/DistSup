import math

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# from src.utils import LReluCustom
from distsup.modules.gan import transformer_encoder
from distsup.modules.gan.data_types import GanConfig


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x *= math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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


        self.positional_embeddings = PositionalEncoding(
            d_model=self.gan_config.dis_emb_size,
            max_len=self.gan_config.max_sentence_length,
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

        y = F.max_pool1d(
            transformer_out.transpose(2, 1),
            kernel_size=self.gan_config.max_sentence_length,
        )

        dense_input = y.view(
            batch_size, self.dense_input_size
        )

        pred = self.dense(dense_input)

        return pred
