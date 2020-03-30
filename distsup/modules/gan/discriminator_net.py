import torch
from torch import nn

# from src.utils import LReluCustom
from distsup.modules.gan.data_types import GanConfig
from distsup.modules.gan.utils import LReluCustom


class DiscriminatorNet(nn.Module):
    def __init__(
        self,
        gan_config: dict,
    ):
        super(DiscriminatorNet, self).__init__()
        self.gan_config = GanConfig(**gan_config)

        self.embedding_matrix = nn.Embedding(
            num_embeddings=self.gan_config.dictionary_size,
            embedding_dim=self.gan_config.dis_emb_size,
        )

        ################################################################
        # Weights initialization with xavier values in embedding matrix
        ################################################################
        nn.init.xavier_uniform_(self.embedding_matrix.weight)
        if self.embedding_matrix.padding_idx is not None:
            with torch.no_grad():
                self.embedding_matrix.weight[
                    self.embedding_matrix.padding_idx
                ].fill_(0)
        ################################################################
        ################################################################

        self.conv_n_feature_1 = self.gan_config.dis_emb_size
        self.conv_3_1 = nn.Conv1d(
            self.conv_n_feature_1,
            self.gan_config.dis_hidden_1_size,
            kernel_size=3,
            padding=1,
        )
        self.conv_5_1 = nn.Conv1d(
            self.conv_n_feature_1,
            self.gan_config.dis_hidden_1_size,
            kernel_size=5,
            padding=2,
        )
        self.conv_7_1 = nn.Conv1d(
            self.conv_n_feature_1,
            self.gan_config.dis_hidden_1_size,
            kernel_size=7,
            padding=3,
        )
        self.conv_9_1 = nn.Conv1d(
            self.conv_n_feature_1,
            self.gan_config.dis_hidden_1_size,
            kernel_size=9,
            padding=4,
        )
        self.lrelu_1 = LReluCustom(leak=0.1)

        self.conv_n_feature_2 = self.gan_config.dis_hidden_1_size * 4
        self.conv_3_2 = nn.Conv1d(
            self.conv_n_feature_2,
            self.gan_config.dis_hidden_2_size,
            kernel_size=3,
            padding=1,
        )
        self.conv_5_2 = nn.Conv1d(
            self.conv_n_feature_2,
            self.gan_config.dis_hidden_2_size,
            kernel_size=3,
            padding=1,
        )
        self.conv_7_2 = nn.Conv1d(
            self.conv_n_feature_2,
            self.gan_config.dis_hidden_2_size,
            kernel_size=3,
            padding=1,
        )
        self.conv_9_2 = nn.Conv1d(
            self.conv_n_feature_2,
            self.gan_config.dis_hidden_2_size,
            kernel_size=3,
            padding=1,
        )

        self.lrelu_2 = LReluCustom(leak=0.1)

        self.dense_input_size = (
            self.gan_config.dis_hidden_2_size * 4
            * self.gan_config.max_sentence_length
        )
        self.dense = nn.Sequential(
            nn.Linear(self.dense_input_size, 1),
        )

    def forward(self, x):
        x = torch.matmul(x, self.embedding_matrix.weight)
        conv1d_input = x.transpose(-1, -2)
        output_c_3_1 = self.conv_3_1(conv1d_input)

        output_c_5_1 = self.conv_5_1(conv1d_input)
        output_c_7_1 = self.conv_7_1(conv1d_input)
        output_c_9_1 = self.conv_9_1(conv1d_input)
        x = torch.cat(
            [output_c_3_1, output_c_5_1, output_c_7_1, output_c_9_1],
            dim=-2
        )
        x = self.lrelu_1(x)

        output_c_3_2 = self.conv_3_2(x)
        output_c_5_2 = self.conv_5_2(x)
        output_c_7_2 = self.conv_7_2(x)
        output_c_9_2 = self.conv_9_2(x)
        x = torch.cat(
            [output_c_3_2, output_c_5_2, output_c_7_2, output_c_9_2],
            dim=-2
        )

        x = self.lrelu_2(x)

        x = x.reshape(-1, self.dense_input_size)
        x = self.dense(x)
        return x
