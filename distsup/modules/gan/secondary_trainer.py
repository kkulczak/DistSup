from copy import deepcopy

from torch import optim
import torch.nn.functional as F
import torch.utils.data

from distsup.configuration import Globals
from distsup.models.GAN_representation_learners import GanRepresentationLearner
from distsup.modules.gan.data_preparation import \
    GanConcatedWindowsDataManipulation
from distsup.modules.gan.data_types import GanConfig
from distsup.modules.gan.utils import (
    assert_as_target,
    assert_one_hot,
    compute_gradient_penalty,
)


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
            gan_config=config,
            encoder_length_reduction=self.model.encoder.length_reduction,
        )

        self.optimizer_gen = optim.Adam(
            self.model.gan_generator.parameters(),
            lr=self.config.gen_learning_rate,
            betas=(0.5, 0.9),
        )
        self.optimizer_dis = optim.Adam(
            self.model.gan_discriminator.parameters(),
            lr=self.config.dis_learning_rate,
            betas=(0.5, 0.9),
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
        _train_bnd, _train_bnd_range, target, _lens = \
            self.data_manipulator.extract_alignment_data(
                batch
            )
        batch = F.one_hot(
            target.long(),
            num_classes=self.config.dictionary_size
        ).float()
        if Globals.cuda:
            batch = batch.to('cuda')
        return batch, target

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
        stats = {}
        for i in range(self.config.dis_steps):
            real_sample, real_target = self.sample_real_batch()
            assert_one_hot(real_sample)
            assert_as_target(real_sample, real_target)

            batched_sample_frame, target, lens = self.sample_gen_batch()
            assert_one_hot(batched_sample_frame)
            assert_as_target(batched_sample_frame, target)

            self.model.gan_discriminator.zero_grad()

            fake_sample = self.model.gan_generator(batched_sample_frame)

            fake_pred = self.model.gan_discriminator(fake_sample)
            real_pred = self.model.gan_discriminator(real_sample)

            fake_score = fake_pred.mean()
            real_score = real_pred.mean()

            gradient_penalty = compute_gradient_penalty(
                self.model.gan_discriminator,
                real_sample,
                fake_sample
            )
            dis_loss = (
                fake_score - real_score
                + self.config.gradient_penalty_ratio * gradient_penalty
            )

            dis_loss.backward()
            self.optimizer_dis.step()

            stats['metrics/gradient_penalty'] = gradient_penalty.item()
            stats['scores/real'] = real_score.item()
            stats['losses/dis'] = dis_loss.item()

        for i in range(self.config.gen_steps):
            batched_sample_frame, target, lens = self.sample_gen_batch()
            assert_one_hot(batched_sample_frame)
            assert_as_target(batched_sample_frame, target)

            self.model.gan_discriminator.zero_grad()

            fake_sample = self.model.gan_generator(batched_sample_frame)

            fake_pred = self.model.gan_discriminator(fake_sample)

            self.model.gan_generator.zero_grad()

            fake_score = fake_pred.mean()
            gen_loss = - fake_score
            gen_loss.backward()
            self.optimizer_gen.step()

            stats['scores/fake'] = fake_score.item()
            stats['losses/gen'] = gen_loss.item()
            stats['scores/diff_abs'] = (
                fake_score.item() - real_score.item()
            )
            stats['gp_impact'] = (
                gradient_penalty.item() / (
                    dis_loss.item()
                ) * self.config.gradient_penalty_ratio
            )
            stats['accuracy/train_batch'] = (
                    target.long() == fake_sample.argmax(-1).long()
            ).float().mean().item()

        return {f'GAN_{k}': v for k, v in stats.items()}
