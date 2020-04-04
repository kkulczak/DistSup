import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distsup import (
    utils,
)
from distsup.logger import default_tensor_logger
from distsup.models import streamtokenizer
from distsup.modules import (
    bottlenecks,
    convolutional,
    encoders,
    reconstructors,
)
from distsup.modules.gan.data_preparation import \
    GanConcatedWindowsDataManipulation
from distsup.modules.gan.data_types import GanConfig
from distsup.modules.gan.utils import (
    assert_one_hot,
    assert_as_target,
)

logger = default_tensor_logger.DefaultTensorLogger()


class GanRepresentationLearner(streamtokenizer.StreamTokenizerNet):
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
        image_height=28,
        in_channels=1,
        encoder=None,
        bottleneck=None,
        bottleneck_latent_dim=64,
        latent_mixer=None,
        reconstructor=None,
        reconstructor_field=None,
        side_info_encoder=None,
        bottleneck_cond=None,
        gan_generator=None,
        gan_discriminator=None,
        **kwargs
    ):
        super(GanRepresentationLearner, self).__init__(**kwargs)
        if encoder is None:
            encoder = dict(
                class_name=convolutional.ConvStack1D,
                hid_channels=64,
                num_strided=2,
                num_dilated=2,
                num_postdil=3
            )
        if bottleneck is None:
            bottleneck = dict(
                class_name=bottlenecks.VQBottleneck,
                num_tokens=16
            )
        if latent_mixer is None:
            latent_mixer = dict(
                class_name=convolutional.ConvStack1D,
                hid_channels=64,
                num_dilated=2,
                num_postdil=2
            )
        if reconstructor is None:
            reconstructor = {
                # name: dict
                '': dict(class_name=reconstructors.ColumnGatedPixelCNN, ),
            }
        self.encoder = utils.construct_from_kwargs(
            encoder, additional_parameters={
                'in_channels': in_channels,
                'image_height': image_height
            })
        # prevent affecting the encoder by the dummy minibatch
        self.encoder.eval()
        enc_out_shape = self.encoder(
            torch.empty((1, 500, image_height, in_channels))).size()

        self.bottleneck = utils.construct_from_kwargs(
            bottleneck, additional_parameters=dict(
                in_dim=enc_out_shape[2] * enc_out_shape[3],
                latent_dim=bottleneck_latent_dim))

        self.latent_mixer = utils.construct_from_kwargs(
            latent_mixer,
            additional_parameters={'in_channels': bottleneck_latent_dim})
        # prevent affecting the latent_mixer by the dummy minibatch
        self.latent_mixer.eval()
        mixer_out_channels = self.latent_mixer(
            torch.empty((1, 500, 1, bottleneck_latent_dim))).size(3)

        cond_channels_spec = [{
            'cond_dim': mixer_out_channels,
            'reduction_factor': self.encoder.length_reduction
        }]

        self.side_info_encoder = None
        if side_info_encoder is not None:
            self.side_info_encoder = utils.construct_from_kwargs(
                side_info_encoder)
            cond_channels_spec.append({
                'cond_dim': side_info_encoder['embedding_dim'],
                'reduction_factor': 0
            })
        self.bottleneck_cond = lambda x: None
        if bottleneck_cond is not None:
            self.bottleneck_cond = utils.construct_from_kwargs(bottleneck_cond)

        # Gan definitions section

        if gan_generator is None:
            self.gan_generator = gan_generator
        else:
            self.gan_generator = utils.construct_from_kwargs(
                gan_generator,
                additional_parameters={
                    'encoder_element_size': self.encoder.hid_channels,
                }
            )
            self.gan_data_manipulator = GanConcatedWindowsDataManipulation(
                gan_config=GanConfig(**gan_generator['gan_config']),
                # encoder_length_reduction=self.encoder.length_reduction,
                encoder_length_reduction=1,
            )
        if gan_discriminator is None:
            self.gan_discriminator = gan_discriminator
        else:
            self.gan_discriminator = utils.construct_from_kwargs(
                gan_discriminator
            )

        rec_params = {
            'image_height': image_height,
            'cond_channels': cond_channels_spec
        }

        # Compatibility with single-reconstructor checkpoints
        if 'class_name' in reconstructor:
            self.reconstructor = utils.construct_from_kwargs(
                reconstructor, additional_parameters=rec_params)
            self.reconstructors = {'': self.reconstructor}
        else:
            self.reconstructors = nn.ModuleDict({
                name: utils.construct_from_kwargs(
                    rec, additional_parameters=rec_params)
                for name, rec in reconstructor.items()})

        if reconstructor_field is None:
            self.reconstructors_fields = [
                'features' for _ in self.reconstructors
            ]

        elif isinstance(reconstructor_field, str):
            self.reconstructors_fields = [
                reconstructor_field for _ in self.reconstructors
            ]

        elif isinstance(reconstructor_field, list):
            self.reconstructors_fields = reconstructor_field

        else:
            raise ValueError(
                f"'reconstructor_field' must be a None, str, or a list. "
                f"Currently {reconstructor_field}"
            )

        assert len(self.reconstructors_fields) == len(self.reconstructors), \
            'The reconstructor_field parameter should have as many elements ' \
            'as reconstructors there are.'

        self.input_layer = encoders.Identity()
        self.add_probes()

    def pad_features(self, batch):
        """Pad x to a multiple of encoder's length reduction so that
           the token stream evenly divides the input.
        """
        if 'features_len' in batch:
            feats = batch['features']
            enc_red = self.encoder.length_reduction
            padded_size = (
                ((batch['features'].size(1) + enc_red - 1) // enc_red) * enc_red
            )
            batch['features'] = F.pad(
                feats,
                [
                    0, 0,  # don't pad C
                    0, 0,  # don't pad H
                    0, padded_size - feats.size(1)  # pad W
                ]
            )
        assert (batch['features'].size(1) % self.encoder.length_reduction) == 0

    def conditioning(self, x, x_len, bottleneck_cond=None):
        """x: N x W x H x C
        """
        x = self.input_layer(x)
        enc = self.encoder(x, x_len)
        enc_len = None
        if x_len is not None:
            enc, enc_len = enc
        try:
            assert enc.shape[1] * self.encoder.length_reduction == x.shape[1]
        except AssertionError as e:
            print(e)
            breakpoint()
        # torch.save(enc, 'enc_out.pt')
        b, t, h, c = enc.size()
        enc = enc.contiguous().view(b, t, 1, h * c)
        quant, kl, info = self.bottleneck(enc, bottleneck_cond, enc_len=enc_len)
        # conds = (self.latent_mixer(quant),)

        return (
            x,
            None,  # conds,
            info,
            enc
        )

    def align_tokens_to_features(self, batch, tokens):
        with torch.no_grad():
            t1 = batch['features'].size(1)
            t2 = tokens.size(1)
            ret = tokens.repeat_interleave((t1 + t2 - 1) // t2, dim=1)
            return ret[:, :t1]

    @staticmethod
    def plot_image_segmentation(fea, recon, indices, recon_name=''):
        # TODO Separate out segmentations (one for all reconstructors)
        #      from reconstructions
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'font.size': 3,
            'xtick.labelsize': 'x-small',
            'ytick.labelsize': 'x-small',
        })
        fea = fea[0, :, :, 0].permute(1, 0)
        recon = recon[0, :, :, 0].permute(1, 0)
        f = plt.Figure(dpi=300)
        ax = f.add_subplot(211)
        ax.imshow(fea.cpu().numpy())
        ax.set_title('orig')
        trans = ax.get_xaxis_transform()
        scale = fea.size(1) // indices.size(1)
        i = 0
        indices = indices.cpu()
        while i < indices.size(1):
            j = i
            while (
                j + 1 < indices.size(1)
                and indices[0, j + 1] == indices[0, i]
            ):
                j += 1
            tok_id = indices[0, j].cpu().item()
            ax.text(j * scale / 2. + i * scale / 2. + scale / 2., -0.01,
                    str(tok_id), rotation=45, transform=trans,
                    horizontalalignment='center', verticalalignment='baseline',
                    color='red')
            ax.axvline(j * scale + scale, linewidth=1.0, linestyle='-')
            i = j + 1
        ax2 = f.add_subplot(212, sharex=ax)
        ax2.imshow(recon.cpu().numpy())
        name = '_' + recon_name if recon_name else recon_name
        logger.log_mpl_figure(f'segmentation{name}', f)

    def reconstruction_loss(self, batch, conds, needs_rec_image):
        if self.side_info_encoder is not None:
            side_info = self.side_info_encoder(batch['side_info'])
            side_info = side_info.unsqueeze(1).unsqueeze(1)
            conds = conds + (side_info,)

        details = {}
        per_pix = {}
        all_inputs = []
        mean_field = []

        for (name, rec), rec_field in zip(self.reconstructors.items(),
                                          self.reconstructors_fields):
            assert rec_field in batch, \
                f"The field to be reconstructed '{rec_field}' not found in " \
                f"batch. Failing."

            assert isinstance(batch[rec_field], torch.Tensor), \
                f"The field '{rec_field}' in the batch is not" \
                f" a torch.Tensor. " \
                f"Possible failing cases:" \
                f"the field has not been added to 'varlen_fields' " \
                f"in the yaml " \
                f"of the PaddedDatasetLoader."

            feats = batch[rec_field]
            feat_lens = batch.get(f'{rec_field}_len')
            if feat_lens is not None:
                def apply_mask(_loss):
                    mask = utils.get_mask1d(
                        feat_lens,
                        mask_length=_loss.size(1)
                    )
                    mask = mask / mask.sum()
                    _loss = _loss * mask.unsqueeze(-1).unsqueeze(-1)
                    height_x_chanels = _loss.size(2) * _loss.size(3)
                    _loss = _loss.sum()
                    # The mask broadcasts over dim 2&3, hence we need to
                    # manually normalize
                    _loss_per_pix = _loss / height_x_chanels
                    return _loss, _loss_per_pix
            else:
                def apply_mask(_loss):
                    _loss = _loss.mean()
                    return _loss, _loss  # nats/pix

            inputs, targets = rec.get_inputs_and_targets(feats, feat_lens)
            logits = rec(inputs, conds)
            loss = rec.loss(logits, targets)
            loss, loss_per_pix = apply_mask(loss)
            name = '_' + name if name else name
            details[f'rec{name}_loss'] = loss
            per_pix[f'rec{name}_loss_per_pix'] = loss_per_pix

            if needs_rec_image:
                all_inputs.append(inputs)
                mean_field.append(rec.get_mean_field_preds(logits.detach()))

        details['rec_loss'] = sum(details.values())
        per_pix['rec_loss_per_pix'] = sum(per_pix.values())
        return (
            details['rec_loss'],
            {**details, **per_pix},
            all_inputs,
            mean_field
        )

    def print_ali_num_segments(self, batch):
        if 'alignment' not in batch:
            return
        seg_idx = batch['alignment']
        num_segments = float(
            (np.diff(seg_idx.view(seg_idx.shape[:2]).cpu().numpy(),
                     axis=1) != 0).sum())
        x_lens = batch.get('features_len', torch.empty(
            batch['alignment'].shape[0], dtype=torch.int64).fill_(
            batch['alignment'].shape[1]))
        print(
            f"Minibatch num segments: {num_segments}, "
            f"enc ratio "
            f"""{
            num_segments / float(x_lens.sum()) *
            self.encoder.length_reduction
            }"""
        )

    def log_images(self, feats, info, inputs, rec_imgs):
        # Note: nn.ModuleDict is ordered
        for (name, rec), rec_input, rec_img in zip(self.reconstructors.items(),
                                                   inputs, rec_imgs):
            if rec_img is None:
                continue

            if info['indices'] is not None:
                self.plot_image_segmentation(
                    feats, rec_img, info['indices'], recon_name=name)

            def log_img(_name, img):
                logger.log_images(_name, img.permute(0, 2, 1, 3))

            name = '_' + name if name else name
            log_img(f'x{name}', rec_input[:4])
            log_img(f'p{name}', rec_img[:4])
            # TODO: make it generate less data
            if False and not self.training:
                raise NotImplementedError('I thinkt tha\'s plots')
                # priming = rec_input[:3].repeat_interleave(2, dim=0)
                # img_conds = [c[:3].repeat_interleave(2, dim=0)
                #     for c in conds]
                # sample_plot = rec.plot_debug_samples(priming, img_conds)
                # if sample_plot is not None:
                #     logger.log_mpl_figure(f'gen_samples{name}', sample_plot)

    def evaluate(self, batches):
        tot_examples = 0.
        # tot_loss = 0.
        # tot_detached_probesloss = 0.
        # tot_backprop_probesloss = 0.
        tot_errs = 0.
        #
        # alis_es = []
        # alis_gt = []
        # alis_lens = []
        # total_stats = {}

        first_batch = None

        for batch in batches:
            if first_batch is None:
                first_batch = copy.deepcopy(batch)

            num_examples = batch['features'].shape[0]
            loss, stats, tokens = self.minibatch_loss_and_tokens(
                batch,
                train_model=False
            )
            literal_errors = [
                (target[:_len] != _tokens[:_len]).sum()
                for _len, target, _tokens in zip(
                    stats['lens'].cpu(),
                    stats['target'].cpu().int(),
                    tokens.cpu().int()
                )
            ]
            tot_errs += sum(literal_errors).item()
            tot_examples += stats['lens'].sum().item()
        print('#' * 40)
        print(stats['target'][0][:stats['lens'][0] + 1])
        print(tokens[0][:stats['lens'][0] + 1])
        print('#' * 40)

        return {
            'literal_errs': tot_errs / tot_examples
        }

        #     if tokens is not None:
        #         # Tokens should be in layout B x W x 1 x 1
        #         tokens = utils.safe_squeeze(tokens, dim=3)
        #         tokens = utils.safe_squeeze(tokens, dim=2)
        #
        #         feat_len = batch['features_len']
        #         alis_lens.append(feat_len)
        #
        #         # the tokens should match the rate of the alignment
        #         ali_es = self.align_tokens_to_features(batch, tokens)
        #         assert (ali_es.shape[0] == batch['features'].shape[0])
        #         assert (ali_es.shape[1] == batch['features'].shape[1])
        #         alis_es.append(ali_es[:, :])
        #         if 'alignment' in batch:
        #             ali_gt = batch['alignment']
        #             ali_len = batch['alignment_len']
        #
        #             assert ((ali_len == feat_len).all())
        #             alis_gt.append(ali_gt)
        #
        #     tot_examples += num_examples
        #     tot_loss += loss * num_examples
        #     tot_errs += stats.get('err', np.nan) * num_examples
        #
        #     tot_detached_probesloss += detached_loss * num_examples
        #     tot_backprop_probesloss += backprop_loss * num_examples
        #     for k, v in stats.items():
        #         if k == 'segmental_values':
        #             if logger.is_currently_logging():
        #                 import matplotlib.pyplot as plt
        #                 f = plt.figure(dpi=300)
        #                 plt.plot(v.data.cpu().numpy(), 'r.-')
        #                 f.set_tight_layout(True)
        #                 logger.log_mpl_figure(f'segmentation_values', f)
        #         elif utils.is_scalar(v):
        #             if k not in total_stats:
        #                 total_stats[k] = v * num_examples
        #             else:
        #                 total_stats[k] += v * num_examples
        # # loss is special, as we use it e.g. for learn rate control
        # # add all signals that we train agains, but remove the passive ones
        # all_scores = {
        #     'loss': (tot_loss + tot_backprop_probesloss) / tot_examples,
        #     'probes_backprop_loss': tot_backprop_probesloss / tot_examples,
        #     'probes_detached_loss': tot_detached_probesloss / tot_examples,
        #     'err': tot_errs / tot_examples,
        #     'probes_loss': (tot_detached_probesloss + tot_backprop_probesloss
        #                     ) / tot_examples
        # }
        #
        # for k, v in total_stats.items():
        #     all_scores[k] = v / tot_examples
        #
        # if (len(alis_es) > 0) and (len(alis_gt) > 0):
        #     # If we have gathered any alignments
        #     f1_scores = dict(precision=[], recall=[], f1=[])
        #     for batch in zip(alis_gt, alis_es, alis_lens):
        #         batch = [t.detach().cpu().numpy() for t in batch]
        #         for k, v in scoring.compute_f1_scores(*batch,
        #         delta=1).items():
        #             f1_scores[k].extend(v)
        #     for k in ('f1', 'precision', 'recall'):
        #         print(f"f1/{k}: {np.mean(f1_scores[k])}")
        #         logger.log_scalar(f'f1/{k}', np.mean(f1_scores[k]))
        #
        #     alis_es = self._unpad_and_concat(alis_es, alis_lens)
        #     alis_gt = self._unpad_and_concat(alis_gt, alis_lens) if len(
        #         alis_gt) else None
        #
        #     scores_to_compute = [('', lambda x: x)]
        #     if alis_gt is not None and self.pad_symbol is not None:
        #         not_pad = (alis_gt != self.pad_symbol)
        #         scores_to_compute.append(('nonpad_', lambda x: x[not_pad]))
        #
        #     if alis_gt is not None and alis_es.min() < 0:
        #         not_pad2 = (alis_es != -1)
        #         scores_to_compute.append(
        #             ('validtokens_', lambda x: x[not_pad2]))
        #
        #     for prefix, ali_filter in scores_to_compute:
        #         es = ali_filter(alis_es)
        #
        #
        #         perplexity_scores = self._perplexity_metrics(es,
        #         prefix=prefix)
        #         all_scores.update(perplexity_scores)
        #
        # return all_scores

    def minibatch_loss_and_tokens(
        self,
        batch,
        train_model=False,
        return_encoder_output=False,
    ):

        # self.print_ali_num_segments(batch)
        self.pad_features(batch)
        feats = batch['features']
        # bottleneck_cond = self.bottleneck_cond(batch)
        # _, conds, info, encoder_output = self.conditioning(
        #     feats,
        #     batch.get('features_len'),
        #     bottleneck_cond
        # )
        info = None
        conds = None
        encoder_output = F.one_hot(
            batch['alignment'].cpu().long(),
            num_classes=self.gan_generator.gan_config.dictionary_size
        ).float().to(batch['features'].device)

        if return_encoder_output:
            return encoder_output.detach()

        if not train_model:
            batched_sample_frame, target, lens = \
                self.gan_data_manipulator.prepare_gan_batch(
                    encoder_output,
                    batch['alignment'].cpu()
                )
            assert_one_hot(batched_sample_frame)
            assert_as_target(batched_sample_frame, target)
            res = self.gan_generator(batched_sample_frame)
            res = res.argmax(dim=-1)
            return (
                0.,
                {
                    'target': target,
                    'lens': lens,
                    'encoder_output': encoder_output,
                    'alignment': batch['alignment'],
                },
                res
            )

        # needs_rec_image = logger.is_currently_logging()
        #
        # rec_loss, details, inputs, rec_imgs = self.reconstruction_loss(
        #     batch, conds, needs_rec_image=needs_rec_image)
        #
        # self.log_images(feats, info, inputs, rec_imgs)

        return (
            torch.tensor(0., requires_grad=True),
            {},
            # details,
            batch['alignment']
        )
