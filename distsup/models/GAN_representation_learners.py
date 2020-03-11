import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from distsup.logger import default_tensor_logger
from distsup.models import streamtokenizer
from distsup.modules import bottlenecks
from distsup.modules import convolutional
from distsup.modules import encoders
from distsup.modules import reconstructors

from distsup import utils

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
                'in_channels':  in_channels,
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
            'cond_dim':         mixer_out_channels,
            'reduction_factor': self.encoder.length_reduction
        }]

        self.side_info_encoder = None
        if side_info_encoder is not None:
            self.side_info_encoder = utils.construct_from_kwargs(
                side_info_encoder)
            cond_channels_spec.append({
                'cond_dim':         side_info_encoder['embedding_dim'],
                'reduction_factor': 0
            })
        self.bottleneck_cond = lambda x: None
        if bottleneck_cond is not None:
            self.bottleneck_cond = utils.construct_from_kwargs(bottleneck_cond)

        rec_params = {
            'image_height':  image_height,
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
        assert enc.shape[1] * self.encoder.length_reduction == x.shape[1]
        b, t, h, c = enc.size()
        enc = enc.contiguous().view(b, t, 1, h * c)
        quant, kl, info = self.bottleneck(enc, bottleneck_cond, enc_len=enc_len)
        conds = (self.latent_mixer(quant),)
        return x, conds, info

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
            'font.size':       3,
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

    def minibatch_loss_and_tokens(self, batch):
        self.print_ali_num_segments(batch)
        self.pad_features(batch)
        feats = batch['features']
        bottleneck_cond = self.bottleneck_cond(batch)
        _, conds, info = self.conditioning(feats, batch.get('features_len'),
                                           bottleneck_cond)
        needs_rec_image = logger.is_currently_logging()

        rec_loss, details, inputs, rec_imgs = self.reconstruction_loss(
            batch, conds, needs_rec_image=needs_rec_image)

        self.log_images(feats, info, inputs, rec_imgs)

        return rec_loss, details, info['indices']

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
