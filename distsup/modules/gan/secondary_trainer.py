from typing import Dict, Tuple

from torch import optim
import torch.nn.functional as F
import torch.utils.data

from distsup.configuration import Globals
from distsup.data import ChunkedDataset
from distsup.models.GAN_representation_learners import GanRepresentationLearner
from distsup.modules.gan.data_preparation import \
    GanConcatedWindowsDataManipulation
from distsup.modules.gan.data_types import EncoderOutput, GanBatch, GanConfig
from distsup.modules.gan.utils import (AlignmentPrettyPrinter, assert_as_target,
                                       assert_one_hot,
                                       compute_gradient_penalty,
                                       generate_alignemnt_dataset, )
from distsup.utils import get_mask1d



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
        self.alignments_dataloader = generate_alignemnt_dataset(train_dataloader)
        self.alignments_iter = iter(self.alignments_dataloader)

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

    def sample_alignments(self):
        try:
            return next(self.alignments_iter)
        except StopIteration:
            self.alignments_iter = iter(self.alignments_dataloader)
            return next(self.alignments_iter)

    def sample_real_batch(self) -> GanBatch:
        batch = self.sample_alignments()
        alignment = batch['alignment']

        real_sample = F.one_hot(
            alignment.long(),
            num_classes=self.config.dictionary_size
        ).float()
        gan_batch = self.data_manipulator.prepare_gan_batch(
            real_sample,
            batch={'alignment': alignment},
            auto_length=False,
            force_single_concat_window=True,
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
        if self.config.supervised:
            sups_stats = self.supervised_train(show=show)
            return sups_stats

        for i in range(self.config.dis_steps):
            self.optimizer_dis.zero_grad()
            real_batch = self.sample_real_batch()
            fake_batch = self.sample_gen_batch()

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
            stats['gan_metrics/real'] = real_score.item()
            stats['loss/gan_discriminator'] = dis_loss.item()

        for i in range(self.config.gen_steps):
            self.optimizer_gen.zero_grad()
            fake_batch = self.sample_gen_batch()

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

            stats['gan_metrics/fake'] = fake_score.item()
            stats['loss/gan_generator'] = gen_loss.item()
            stats['gan_metrics/diff_abs'] = (
                fake_score.item() - real_score.item()
            )
            stats['gan_metrics/gp_impact'] = (
                gradient_penalty.item() / (
                dis_loss.item()
            ) * self.config.gradient_penalty_ratio
            )

        if self.config.gen_steps != 0:
            mask = get_mask1d(
                fake_batch.lens,
                self.config.max_sentence_length
            ).to(torch.bool)
            corrects = (fake_batch.target.long() == fake_sample.argmax(-1).long(
            )).float()
            stats['acc/gan_train_batch'] = corrects[mask].mean().item()

            if show:
                for i in range(1):
                    self.printer.show(
                        fake_sample[i].argmax(-1).long(),
                        fake_batch.target[i].long()
                    )
                print('#' * self.printer.line_length)

        return stats

    def supervised_train(self, show=False):
        stats = {}
        self.model.gan_generator.zero_grad()
        fake_batch = self.sample_gen_batch()

        fake_sample = self.model.gan_generator(fake_batch.data)
        if self.config.use_all_letters:
            fake_sample = fake_sample.repeat_interleave(
                self.model.encoder.length_reduction,
                dim=1
            )

        losses = F.nll_loss(
            fake_sample.permute(0, 2, 1),
            fake_batch.target.long(),
            reduction='none',
        )
        mask = get_mask1d(
            fake_batch.lens,
            self.config.max_sentence_length
        )
        mask = mask / mask.sum()
        loss = (losses * mask).sum()
        loss.backward()
        self.optimizer_gen.step()

        pred_labels = fake_sample.argmax(dim=2).long()
        acc = ((pred_labels == fake_batch.target).float() * mask).sum()

        stats['loss/gan_probe'] = loss.item()
        stats['acc/gan_probe_batch'] = acc.item()

        if show:
            for i in range(1):
                self.printer.show(
                    fake_sample[i].argmax(-1).long(),
                    fake_batch.target[i].long()
                )
            print('#' * self.printer.line_length)

        return stats
