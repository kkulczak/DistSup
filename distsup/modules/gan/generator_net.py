from torch import nn

from distsup.modules.gan.data_types import GanConfig
from distsup.modules.gan.utils import softmax_gumbel_noise


class LinearGeneratorNet(nn.Module):
    gan_config: GanConfig

    def __init__(
        self,
        gan_config: dict,
        encoder_element_size: int,
        z_size: int = 0,
        **kwargs,
    ):
        super(LinearGeneratorNet, self).__init__()
        self.gan_config = GanConfig(**gan_config)
        self.encoder_element_size = encoder_element_size
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

        return soft_prob
