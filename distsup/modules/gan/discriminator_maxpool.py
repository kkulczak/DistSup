import math

import torch
from torch import nn

# from src.utils import LReluCustom
from distsup.modules.gan import utils
from distsup.modules.gan.data_types import GanConfig


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


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

        num_maxpools = int(math.log2(self.gan_config.dis_maxpool_reduction))

        def conv(in_channels, out_channels):
            return nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )

        channels = [
            (self.gan_config.dis_emb_size, self.gan_config.dis_hidden_1_size)
        ]

        reducing_modules = [
            conv(*channels[0]),
            utils.LReluCustom(),
        ]

        channels_per_module = next_power_of_2(
            self.gan_config.dis_hidden_2_size // num_maxpools
        )
        for i in range(num_maxpools):
            ch = (channels[-1][1], channels_per_module)
            reducing_modules.extend([
                nn.MaxPool1d(2, stride=2),
                conv(*ch),
                utils.LReluCustom(),
            ])
            channels.append(ch)

        self.reducing_convs = nn.Sequential(*reducing_modules)
        self.dense_input_size = (
            channels_per_module
            * self.gan_config.max_sentence_length
            // self.gan_config.dis_maxpool_reduction
        )
        self.dense = nn.Linear(self.dense_input_size, 1)

    def forward(self, x):
        batch_size, phrase_length, element_size = x.shape
        x = torch.matmul(x, self.embeddings.weight)

        conv1d_input = x.transpose(2, 1)

        reduced_conv = self.reducing_convs(conv1d_input)

        dense_input = reduced_conv.view(
            batch_size, self.dense_input_size
        )

        pred = self.dense(dense_input)

        return pred
