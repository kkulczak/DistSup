import torch
from torch import nn

# from src.utils import LReluCustom
from distsup.modules.gan import utils
from distsup.modules.gan.data_types import GanConfig


class MaxPoolDiscriminator(nn.Module):
    def __init__(
        self,
        gan_config: dict,
    ):
        super(MaxPoolDiscriminator, self).__init__()
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

        # self.convs1 = nn.ModuleList([
        #     nn.Conv1d(
        #         in_channels=self.conv_n_feature_1,
        #         out_channels=self.gan_config.dis_hidden_1_size,
        #         kernel_size=kernel_size,
        #         padding=padding,
        #     )
        #     for (kernel_size, padding) in [(3, 1), (5, 2), (7, 3), (9, 4)]
        #
        # ])

        self.num_reducing_layers = 4
        dim = 512
        self.reducing_convs = nn.Sequential(
            nn.Conv1d(
                in_channels=self.gan_config.dis_emb_size,
                out_channels=1024,
                kernel_size=3,
                padding=1,
            ),
            utils.LReluCustom(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(
                in_channels=1024,
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            utils.LReluCustom(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            utils.LReluCustom(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            utils.LReluCustom(),
        )
        self.dense_input_size = (
            512 *
            self.gan_config.max_sentence_length // (2 ** 3)
        )
        self.dense = nn.Linear(self.dense_input_size, 1)


    def forward(self, x):
        batch_size, phrase_length, element_size = x.shape
        x = torch.matmul(x, self.embeddings.weight)

        conv1d_input = x.transpose(2, 1)
        # convs_1 = [utils.lrelu(conv(conv1d_input)) for conv in self.convs1]
        # convs_1_out = torch.cat(
        #     convs_1,
        #     dim=1,
        # )

        reduced_conv = self.reducing_convs(conv1d_input)

        dense_input = reduced_conv.view(
            batch_size, self.dense_input_size
        )

        pred = self.dense(dense_input)

        return pred
