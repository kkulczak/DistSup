import torch
from torch import nn

# from src.utils import LReluCustom
from distsup.modules.gan import transformer_encoder
from distsup.modules.gan.data_types import GanConfig


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

        self.transformer = transformer_encoder.TransformerModel(
            ntoken=self.gan_config.dictionary_size,
            ninp=self.gan_config.dis_emb_size,
            # nhead=2,
            nhid=self.gan_config.dis_hidden_1_size,
            # nlayers=2,
            # dropout=0.2,
        )

        self.dense_input_size = (
            self.gan_config.dis_emb_size
        )
        self.dense = nn.Linear(self.dense_input_size, 1)

    def forward(self, x):
        batch_size, phrase_length, element_size = x.shape
        x = torch.matmul(x, self.embeddings.weight)

        transformer_out = self.transformer(x)

        y = transformer_out[:, -1, :]

        dense_input = y.view(
            batch_size, self.dense_input_size
        )

        pred = self.dense(dense_input)

        return pred
