import torch
from torch import nn

from distsup.modules.gan.utils import softmax_gumbel_noise


class LinearGeneratorNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        concat_window: int,
        encoder_element_size: int,
        dictionary_size: int,
        # phrase_length: int,
        z_size: int = 0,
    ):
        super(LinearGeneratorNet, self).__init__()

        self.hidden_size = hidden_size
        self.concat_window = concat_window
        self.encoder_element_size = encoder_element_size
        # self.phrase_length = phrase_length
        self.dictionary_size = dictionary_size
        self.z_size = z_size

        self.n_feature = (
            self.concat_window * self.encoder_element_size
            + self.z_size
        )
        self.n_out = self.dictionary_size

        self.hidden1 = nn.Sequential(
            nn.Linear(
                self.n_feature,
                self.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n_out),
        )

    def forward(self, x, temperature: float = 0.9):
        batch_size, phrase_length, _ = x.shape
        x = x.reshape(
            batch_size * phrase_length,
            self.concat_window * self.encoder_element_size
        )

        #NOT WORKING Z RANDOM SEED INJECTION
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
            self.dictionary_size
        )
        soft_prob = softmax_gumbel_noise(log_prob, temperature)

        return soft_prob
