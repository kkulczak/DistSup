from typing import Dict, Tuple

from torch import optim
import torch.nn.functional as F
import torch.utils.data

from distsup.configuration import Globals
from distsup.models.GAN_representation_learners import GanRepresentationLearner
from distsup.modules.gan.data_preparation import \
    GanConcatedWindowsDataManipulation
from distsup.modules.gan.data_types import EncoderOutput, GanBatch, GanConfig
from distsup.modules.gan.utils import (AlignmentPrettyPrinter, assert_as_target,
                                       assert_one_hot,
                                       compute_gradient_penalty, )


class SecondaryTrainerGAN:
    config: GanConfig

    def __init__(
        self,
        model: GanRepresentationLearner,
        train_dataloader: torch.utils.data.DataLoader,
        config: GanConfig,
    ):

        self.model = model
        # self.vanilla_dataloader = deepcopy(train_dataloader)
        self.vanilla_dataloader = train_dataloader
        self.dataloader_iter = iter(self.vanilla_dataloader)
        self.config = config
        self.data_manipulator = GanConcatedWindowsDataManipulation(
            gan_config=config,
            encoder_length_reduction=self.model.encoder.length_reduction,
        )
        if Globals.debug:
            self.alignments = None
        else:
            self.alignments = torch.cat(
                [batch['alignment'] for batch in train_dataloader],
                dim=0,
            )
            self.gan_alignments = self.data_manipulator.extract_alignment_data(
                self.alignments
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
        self.printer = AlignmentPrettyPrinter(dataloader=train_dataloader)

    def sample_vanilla_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.vanilla_dataloader)
            return next(self.dataloader_iter)

    def sample_real_batch(self) -> GanBatch:
        if Globals.debug:
            batch = self.sample_vanilla_batch()
            alignment = batch['alignment']
        else:
            batch = {}
            idx = torch.randint(
                high=self.alignments.shape[0],
                size=(self.vanilla_dataloader.batch_size,),
            )
            alignment = self.alignments[idx]

        real_sample = F.one_hot(
            alignment.long(),
            num_classes=self.config.dictionary_size
        ).float()
        gan_batch = self.data_manipulator.prepare_gan_batch(
            real_sample,
            batch={'alignment': alignment},
            auto_length=False,
        )

        if Globals.cuda:
            gan_batch = self.data_manipulator.to_cuda(gan_batch)
        return gan_batch

    def sample_batch_from_encoder(self
    ) -> Tuple[EncoderOutput, Dict[str, torch.Tensor]]:
        batch = self.sample_vanilla_batch()
        if Globals.cuda:
            batch = self.model.batch_to_device(batch, 'cuda')
        encoder_output = self.model.minibatch_loss_and_tokens(
            batch,
            return_encoder_output=True
        )
        return encoder_output, batch

    def sample_gen_batch(self) -> GanBatch:
        encoder_output, batch = self.sample_batch_from_encoder()
        gan_batch = self.data_manipulator.prepare_gan_batch(
            encoder_output.data,
            batch,
            auto_length=False,
        )
        if Globals.cuda:
            gan_batch = self.data_manipulator.to_cuda(gan_batch)
        return gan_batch

    def iterate_step(self, show=False):
        stats = {}
        for i in range(self.config.dis_steps):
            self.model.gan_discriminator.zero_grad()
            real_batch = self.sample_real_batch()
            if self.model.encoder.identity:
                assert_one_hot(real_batch.data)
                assert_as_target(real_batch.data, real_batch.target)

            fake_batch = self.sample_gen_batch()
            if self.model.encoder.identity:
                assert_one_hot(fake_batch.data)
                assert_as_target(fake_batch.data, fake_batch.target)

            fake_sample = self.model.gan_generator(fake_batch.data)

            if self.config.use_all_letters:
                fake_sample = fake_sample.repeat_interleave(
                    self.model.encoder.length_reduction,
                    dim=1
                )

            fake_pred = self.model.gan_discriminator(fake_sample)
            real_pred = self.model.gan_discriminator(real_batch.data)

            fake_score = fake_pred.mean()
            real_score = real_pred.mean()

            gradient_penalty = compute_gradient_penalty(
                self.model.gan_discriminator,
                real_batch.data,
                fake_sample
            )
            dis_loss = (
                fake_score - real_score
                + self.config.gradient_penalty_ratio * gradient_penalty
            )

            dis_loss.backward()
            self.optimizer_dis.step()

            stats['gan_metrics/gradient_penalty'] = gradient_penalty.item()
            stats['gan_scores/real'] = real_score.item()
            stats['gan_losses/dis'] = dis_loss.item()

        for i in range(self.config.gen_steps):
            self.model.gan_generator.zero_grad()
            fake_batch = self.sample_gen_batch()
            if self.model.encoder.identity:
                assert_one_hot(fake_batch.data)
                assert_as_target(fake_batch.data, fake_batch.target)

            fake_sample = self.model.gan_generator(fake_batch.data)
            if self.config.use_all_letters:
                fake_sample = fake_sample.repeat_interleave(
                    self.model.encoder.length_reduction,
                    dim=1
                )

            fake_pred = self.model.gan_discriminator(fake_sample)

            fake_score = fake_pred.mean()
            gen_loss = - fake_score
            gen_loss.backward()
            self.optimizer_gen.step()

            stats['gan_scores/fake'] = fake_score.item()
            stats['gan_losses/gen'] = gen_loss.item()
            stats['gan_scores/diff_abs'] = (
                fake_score.item() - real_score.item()
            )
            stats['gp_impact'] = (
                gradient_penalty.item() / (
                dis_loss.item()
            ) * self.config.gradient_penalty_ratio
            )

            # mask = get_mask1d(
            #     lens,
            #     self.config.max_sentence_length
            # ).to(torch.bool)
            # corrects = (target.long() == fake_sample.argmax(-1).long(
            # )).float()
            # stats['gan_accuracy/acc'] = corrects[mask].mean().item()
            # stats[
            #     'gan_accuracy/acc_without_mask'
            # ] = corrects.mean().item()
            # stats['gan_accuracy/mask_coverage'] = mask.float().mean().item()

        if show:
            for i in range(1):
                self.printer.show(
                    fake_sample[i].argmax(-1).long(),
                    fake_batch.target[i].long()
                )
            print('#' * self.printer.line_length)

        return stats
