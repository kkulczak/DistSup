import torch
import torch.nn as nn
import torch.nn.functional as F

from distsup.modules.gan.data_types import GanConfig
from distsup.modules.gan.utils import softmax_gumbel_noise


class LinearGeneratorNet(nn.Module):
    gan_config: GanConfig

    def __init__(
        self,
        gan_config: dict,
        encoder_element_size: int,
        encoder_length_reduction: int,
        z_size: int = 0,
        **kwargs,
    ):
        super(LinearGeneratorNet, self).__init__()
        self.gan_config = GanConfig(**gan_config)
        self.encoder_element_size = encoder_element_size
        self.encoder_length_reduction = encoder_length_reduction
        # self.phrase_length = phrase_length
        self.z_size = z_size

        self.n_feature = (
            self.gan_config.concat_window * self.encoder_element_size
            + self.z_size
        )
        self.n_out = self.gan_config.dictionary_size

        self.hidden1 = nn.Sequential(
            nn.Linear(
                self.n_feature,
                self.gan_config.gen_hidden_size
            ),
            nn.ReLU(),
            nn.Linear(self.gan_config.gen_hidden_size, self.n_out),
        )

    def forward(self, x: torch.Tensor, temperature: float = 0.9):
        # x = safe_squeeze(x, dim=2)
        batch_size, phrase_length, _ = x.shape
        x = x.reshape(
            batch_size * phrase_length,
            self.gan_config.concat_window * self.encoder_element_size
        )

        # NOT WORKING Z RANDOM SEED INJECTION
        # if self.z_size > 0:
        #     generator_seed = torch.rand(
        #         (batch_size, self.z_size),
        #         dtype=torch.float32,
        #         device=x.device,
        #     ).repeat_interleave(repeats=self.config['phrase_length'], dim=0)
        #     x = torch.cat([generator_seed, x], dim=1)

        x = self.hidden1(x)

        log_prob = x.reshape(
            batch_size,
            phrase_length,
            self.gan_config.dictionary_size
        )
        soft_prob = softmax_gumbel_noise(log_prob, temperature)

        # soft_prob = F.softmax(log_prob, dim=-1)
        return soft_prob


class ConvGeneratorNet(nn.Module):
    gan_config: GanConfig

    def __init__(
        self,
        gan_config: dict,
        encoder_element_size: int,
        encoder_length_reduction: int,
        z_size: int = 0,
        **kwargs,
    ):
        super(ConvGeneratorNet, self).__init__()
        self.gan_config = GanConfig(**gan_config)
        self.encoder_element_size = encoder_element_size
        self.encoder_length_reduction = encoder_length_reduction

        self.n_feature = (
            self.gan_config.concat_window * self.encoder_element_size
        )
        self.n_out = self.gan_config.dictionary_size

        self.conv = nn.Sequential(
            nn.Conv1d(
                self.n_feature,
                self.gan_config.gen_hidden_size,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(self.gan_config.gen_hidden_size, self.n_out),
        )

    def forward(self, x: torch.Tensor, temperature: float = 0.9):
        # x = safe_squeeze(x, dim=2)
        batch_size, phrase_length, element_size = x.shape
        x = x.permute(0, 2, 1)

        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * phrase_length,
                      self.gan_config.gen_hidden_size)

        x = self.linear(x)

        log_prob = x.reshape(
            batch_size,
            phrase_length,
            self.gan_config.dictionary_size
        )
        soft_prob = softmax_gumbel_noise(log_prob, temperature)

        return soft_prob


class OneConv(nn.Module):
    gan_config: GanConfig

    def __init__(
        self,
        gan_config: dict,
        encoder_element_size: int,
        activation: str = 'gumbel',
        **kwargs,
    ):
        super(OneConv, self).__init__()
        self.gan_config = GanConfig(**gan_config)
        self.pred = nn.Conv1d(
            encoder_element_size,
            self.gan_config.dictionary_size,
            kernel_size=3,
            padding=1,
        )
        if activation == 'gumbel':
            self.activation = lambda x: softmax_gumbel_noise(x, 0.9)
        elif activation == 'softmax':
            self.activation = lambda x: F.softmax(x, dim=-1)
        elif activation == 'none':
            self.activation = lambda x: x
        else:
            raise NotImplemented(f'activation ({activation}) is not defined')


    def forward(self, x: torch.Tensor):
        x = x.transpose(2, 1)
        x = self.pred(x)
        x = x.transpose(2, 1)
        return self.activation(x)
