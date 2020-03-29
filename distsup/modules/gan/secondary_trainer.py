from copy import deepcopy
from dataclasses import dataclass

import torch.utils.data
from torch import nn

from distsup.configuration import Globals
from distsup.models.GAN_representation_learners import GanRepresentationLearner
from distsup.modules.gan.data_preparation import \
    GanConcatedWindowsDataManipulation



@dataclass
class GanConfig:
    concat_window: int
    dictionary_size: int
    max_sentence_length: int
    repeat: int
    dis_steps: int
    gen_hidden_size: int
    gen_steps: int


class SecondaryTrainerGAN:
    config: GanConfig

    def __init__(
        self,
        model: GanRepresentationLearner,
        train_dataloader: torch.utils.data.DataLoader,
        config: GanConfig,
    ):

        self.model = model
        self.vanilla_dataloader = deepcopy(train_dataloader)
        self.dataloader_iter = iter(self.vanilla_dataloader)
        self.config = config
        self.data_manipulator = GanConcatedWindowsDataManipulation(
            encoder_length_reduction=self.model.encoder.length_reduction,
            concat_window=config.concat_window,
            max_sentence_length=config.max_sentence_length,
            repeat=config.repeat
        )

    def sample_vanilla_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.vanilla_dataloader)
            return next(self.dataloader_iter)

    def sample_real_batch(self, device: str = 'cpu'):
        alignments = [
            self.sample_vanilla_batch()['alignment']
            for _ in range(self.config.repeat)
        ]
        batch = torch.cat(alignments, dim=0)
        if Globals.cuda:
            batch = batch.to(device)
        return batch

    def sample_batch_from_encoder(self):
        batch = self.sample_vanilla_batch()
        if Globals.cuda:
            batch = self.model.batch_to_device(batch, 'cuda')
        encoder_output = self.model.minibatch_loss_and_tokens(
            batch,
            return_encoder_output=True
        )
        return encoder_output, batch['alignment']

    def sample_gen_batch(self):
        encoder_output, alignment = self.sample_batch_from_encoder()
        batched_sample_frame, target, lens = \
            self.data_manipulator.prepare_gan_batch(
                encoder_output.cpu(),
                alignment.cpu()
        )
        if Globals.cuda:
            batched_sample_frame = batched_sample_frame.to('cuda')
            target = target.to('cuda')
            lens = lens.to('cuda')

        return batched_sample_frame, target, lens

    def iterate_step(self):
        for i in range(self.config.dis_steps):
            real_batch = self.sample_real_batch()
            batched_sample_frame, target, lens = self.sample_gen_batch()
            fake_sample = self.model.gan_generator(batched_sample_frame)
        for i in range(self.config.gen_steps):
            pass