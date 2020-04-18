import copy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distsup import (
    utils,
    scoring, )
from distsup.logger import default_tensor_logger
from distsup.models import streamtokenizer
from distsup.models.representation_learners import RepresentationLearner
from distsup.modules import (
    bottlenecks,
    convolutional,
    encoders,
    reconstructors,
)
from distsup.modules.gan.data_preparation import \
    GanConcatedWindowsDataManipulation
from distsup.modules.gan.data_types import EncoderOutput, GanConfig
from distsup.modules.gan.utils import assert_one_hot, assert_as_target
from distsup.utils import get_mask1d, safe_squeeze

logger = default_tensor_logger.DefaultTensorLogger()


class GanRepresentationLearner(RepresentationLearner):
    """A basic representation learner.

    The data goes through:
    1. encoder (extracts letent representation)
    2. bottleneck
    3. latent_mixer mixes neighboring latent vectors
    4. reconstructor (reconstructs the inputs in an autoregressive or
                      deconvolutional or other way)
    """

    def __init__(
        self,
        gan_generator=None,
        gan_discriminator=None,
        **kwargs
    ):
        super(GanRepresentationLearner, self).__init__(
            add_probes=False,
            **kwargs
        )

        # Gan definitions section
        if gan_generator is None:
            self.gan_generator = gan_generator
        else:
            self.gan_generator = utils.construct_from_kwargs(
                gan_generator,
                additional_parameters={
                    'encoder_element_size': self.encoder.hid_channels,
                    'encoder_length_reduction': self.encoder.length_reduction,
                }
            )
            self.gan_data_manipulator = GanConcatedWindowsDataManipulation(
                gan_config=GanConfig(**gan_generator['gan_config']),
                encoder_length_reduction=self.encoder.length_reduction,
            )
        if gan_discriminator is None:
            self.gan_discriminator = gan_discriminator
        else:
            self.gan_discriminator = utils.construct_from_kwargs(
                gan_discriminator
            )

        self.printer = None
        self.add_probes()

    def deprecated_evaluate(self, batches):
        probe_acc = []
        acc = []
        acc_no_padding = []
        mask_coverage = []
        first_batch = None
        for batch in batches:
            if first_batch is None:
                first_batch = copy.deepcopy(batch)

            loss, stats, torch_tokens = self.minibatch_loss_and_tokens(
                batch,
                train_model=False
            )
            if self.gan_data_manipulator.use_all_letters:
                lens: np.ndarray = batch['alignment_len'].cpu().numpy()
            else:
                lens: np.ndarray = stats['lens'].cpu().numpy()
            target: np.ndarray = stats['target'].cpu().int().numpy()
            tokens: np.ndarray = (
                torch_tokens.cpu().int().numpy()[:, :target.shape[1]]
            )
            mask = get_mask1d(
                torch.from_numpy(lens),
                mask_length=target.shape[1]
            ).to(torch.bool).numpy()
            correct = (tokens == target)
            acc.append(correct[mask].mean())
            acc_no_padding.append(correct.mean())
            mask_coverage.append(mask.mean())

            # probe stats
            if 'enc_sup' in self.probes.keys():
                encoder_output = stats['encoder_output']
                probe_pred = safe_squeeze(
                    self.probes['enc_sup'](encoder_output.data),
                    dim=2,
                )
                res_pred = probe_pred.repeat_interleave(
                    self.encoder.length_reduction,
                    dim=1
                )
                probe_tokens = res_pred.argmax(dim=2)[:, :target.shape[1]]
                probe_acc.append(
                    (target == probe_tokens.cpu().numpy())[mask].mean()
                )

        for i in range(3):
            self.printer.show(
                torch.from_numpy(tokens[i]),
                stats['target'][i],
            )
        print('#' * self.printer.line_length)
        return {
            'gan_accuracy/acc': np.array(acc).mean(),
            'gan_accuracy/acc_without_mask': np.array(acc_no_padding).mean(),
            'gan_accuracy/probe': np.array(probe_acc).mean(),
            'gan_accuracy/mask_coverage': np.array(mask_coverage).mean()

        }

    def conditioning(self, x, x_len, bottleneck_cond=None):
        """x: N x W x H x C
        """
        x = self.input_layer(x)
        enc = self.encoder(x, x_len)
        enc_len = None
        if x_len is not None:
            enc, enc_len = enc
        assert enc.shape[1] * self.encoder.length_reduction == x.shape[1]
        b, t, h, c = enc.size()
        enc = enc.contiguous().view(b, t, 1, h * c)
        quant, kl, info = self.bottleneck(enc, bottleneck_cond, enc_len=enc_len)
        conds = (self.latent_mixer(quant),)
        return (
            EncoderOutput(data=enc, lens=enc_len),
            tuple(),    # conds,
            {},     # info
        )

    def minibatch_loss_and_tokens(
        self,
        batch,
        train_model=False,
        return_encoder_output=False,
    ):

        self.pad_features(batch)
        feats = batch['features']
        encoder_output, conds, info = self.conditioning(
            feats,
            batch.get('features_len'),
            self.bottleneck_cond(batch)
        )
        if self.encoder.identity:
            encoder_output = EncoderOutput(
                data=F.one_hot(
                    batch['alignment'].long(),
                    num_classes=self.gan_generator.gan_config.dictionary_size
                ).unsqueeze(dim=2).float(),
                lens=batch.get('alignment_len')
            )

        if return_encoder_output:
            return EncoderOutput(
                data=encoder_output.data.detach(),
                lens=encoder_output.lens,
            )

        # needs_rec_image = logger.is_currently_logging()
        # rec_loss, details, inputs, rec_imgs = self.reconstruction_loss(
        #     batch, conds, needs_rec_image=needs_rec_image and False)
        # if needs_rec_image and False:
        #     self.log_images(feats, info, inputs, rec_imgs)

        if self.encoder.identity:
            assert_one_hot(encoder_output.data)
            assert_as_target(encoder_output.data, batch['alignment'])

        gen_output: torch.Tensor = self.gan_generator(encoder_output.data)
        tokens = gen_output.argmax(dim=-1).unsqueeze(dim=2).unsqueeze(dim=3)
        return (
            torch.tensor(0., requires_grad=True, device=feats.device),   #rec_loss,
            {},                                     #details,
            tokens
        )
