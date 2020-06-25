import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from distsup import (
    scoring, utils,
)
from distsup.logger import default_tensor_logger
from distsup.models.representation_learners import RepresentationLearner
from distsup.modules.gan.alignment_shuffler import AlignmentShuffler
from distsup.modules.gan.data_preparation import \
    (GanConcatedWindowsDataManipulation)
from distsup.modules.gan.data_types import EncoderOutput, GanConfig
from distsup.modules.gan.utils import (assert_as_target, assert_one_hot,
                                       EncoderTokensProtos, )

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
    letters_protos: Optional[EncoderTokensProtos] = None
    alignment_shuffler: Optional[AlignmentShuffler] = None

    def __init__(
        self,
        gan_generator=None,
        gan_discriminator=None,
        letters_protos=None,
        alignment_shuffler=None,
        **kwargs
    ):
        super(GanRepresentationLearner, self).__init__(
            add_probes=False,
            **kwargs
        )

        # Gan definitions section
        self.gan_config = GanConfig(**gan_generator['gan_config'])
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
                gan_config=self.gan_config,
                encoder_length_reduction=self.encoder.length_reduction,
            )
        if gan_discriminator is None:
            self.gan_discriminator = gan_discriminator
        else:
            self.gan_discriminator = utils.construct_from_kwargs(
                gan_discriminator
            )
        if letters_protos is not None:
            self.letters_protos = utils.construct_from_kwargs(
                letters_protos,
                additional_parameters={
                    'num_tokens': self.gan_config.dictionary_size
                }
            )
        if alignment_shuffler is not None:
            self.alignment_shuffler = utils.construct_from_kwargs(
                alignment_shuffler,
                additional_parameters={
                    'dict_size': self.gan_config.dictionary_size
                }
            )


        self.printer = None
        self.add_probes()
        print(self.gan_generator)
        print(self.gan_discriminator)
        print(
            f'Dis length reduction: {self.gan_config.max_sentence_length} => '
            f'''{
            self.gan_config.max_sentence_length
            // self.gan_config.dis_maxpool_reduction
            }'''
        )

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
            tuple(),  # conds,
            {},  # info
        )

    def evaluate(self, batches):
        tot_examples = 0.
        tot_loss = 0.
        tot_detached_probesloss = 0.
        tot_backprop_probesloss = 0.
        tot_errs = 0.

        gan_es = []
        gan_gt = []
        gan_lens = []

        probe_enc_sup_es = []

        alis_es = []
        alis_gt = []
        alis_lens = []
        total_stats = {}

        first_batch = None

        for batch in batches:
            if first_batch is None:
                first_batch = copy.deepcopy(batch)

            num_examples = batch['features'].shape[0]
            loss, stats, tokens = self.minibatch_loss_and_tokens(batch)

            # Run the probes
            detached_loss, backprop_loss, probes_details = self.probes_loss(
                batch)
            stats.update(probes_details)
            ### AWEFULL INJECTION
            # if tokens.squeeze().shape != probes_details[
            # 'enc_sup_out_seq'].shape:
            #     print(
            #         tokens.squeeze().shape,
            #         probes_details['enc_sup_out_seq'].shape
            #     )
            #     breakpoint()
            ## INJECTION
            if 'enc_sup_out_seq' in probes_details:
                probe_enc_sup_es.append(probes_details['enc_sup_out_seq'])
            # tokens = probes_details['enc_sup_tokens']

            if (
                'gan_tokens' in stats
                and 'gan_batch' in stats
                and not self.gan_config.use_all_letters
            ):
                gan_es.append(stats['gan_tokens'])
                gan_gt.append(stats['gan_batch'].target)
                gan_lens.append(stats['gan_batch'].lens)

            if tokens is not None:
                # Tokens should be in layout B x W x 1 x 1
                tokens = utils.safe_squeeze(tokens, dim=3)
                tokens = utils.safe_squeeze(tokens, dim=2)

                feat_len = batch['features_len']
                alis_lens.append(feat_len)

                # the tokens should match the rate of the alignment
                ali_es = self.align_tokens_to_features(batch, tokens)
                assert (ali_es.shape[0] == batch['features'].shape[0])
                assert (ali_es.shape[1] == batch['features'].shape[1])
                alis_es.append(ali_es[:, :])
                if 'alignment' in batch:
                    ali_gt = batch['alignment']
                    ali_len = batch['alignment_len']

                    assert ((ali_len == feat_len).all())
                    alis_gt.append(ali_gt)

            tot_examples += num_examples
            tot_loss += loss * num_examples
            tot_errs += stats.get('err', np.nan) * num_examples

            tot_detached_probesloss += detached_loss * num_examples
            tot_backprop_probesloss += backprop_loss * num_examples
            for k, v in stats.items():
                if k == 'segmental_values':
                    if logger.is_currently_logging():
                        import matplotlib.pyplot as plt
                        f = plt.figure(dpi=300)
                        plt.plot(v.data.cpu().numpy(), 'r.-')
                        f.set_tight_layout(True)
                        logger.log_mpl_figure(f'segmentation_values', f)
                elif utils.is_scalar(v):
                    if k not in total_stats:
                        total_stats[k] = v * num_examples
                    else:
                        total_stats[k] += v * num_examples
        # loss is special, as we use it e.g. for learn rate control
        # add all signals that we train agains, but remove the passive ones
        all_scores = {
            'loss': (tot_loss + tot_backprop_probesloss) / tot_examples,
            'probes_backprop_loss': tot_backprop_probesloss / tot_examples,
            'probes_detached_loss': tot_detached_probesloss / tot_examples,
            'err': tot_errs / tot_examples,
            'probes_loss': (tot_detached_probesloss + tot_backprop_probesloss
                            ) / tot_examples
        }

        for k, v in total_stats.items():
            all_scores[k] = v / tot_examples

        if (len(alis_es) > 0) and (len(alis_gt) > 0):
            # If we have gathered any alignments
            # f1_scores = dict(precision=[], recall=[], f1=[])
            # for batch in zip(alis_gt, alis_es, alis_lens):
            #     batch = [t.detach().cpu().numpy() for t in batch]
            #     for k, v in scoring.compute_f1_scores(*batch, delta=1).items():
            #         f1_scores[k].extend(v)
            # for k in ('f1', 'precision', 'recall'):
            #     print(f"f1/{k}: {np.mean(f1_scores[k])}")
            #     logger.log_scalar(f'f1/{k}', np.mean(f1_scores[k]))

            alis_es = self._unpad_and_concat(alis_es, alis_lens)
            # alis_gt = self._unpad_and_concat(alis_gt, alis_lens) if len(
            #     alis_gt) else None

            scores_to_compute = [
                #     {
                #     'prefix': 'all',
                #     'es': alis_es,
                #     'gt': alis_gt,
                # }
            ]
            # if alis_gt is not None and self.pad_symbol is not None:
            #     not_pad = (alis_gt != self.pad_symbol)
            #     scores_to_compute.append({
            #         'prefix': 'nonpad',
            #         'es': alis_es[not_pad],
            #         'gt': alis_gt[not_pad]
            #     })
            if len(gan_es) > 0 and len(gan_gt) > 0:
                gan_es = self._unpad_and_concat(gan_es, gan_lens)
                gan_gt = self._unpad_and_concat(gan_gt, gan_lens)
                scores_to_compute.append({
                    'prefix': 'gan_tokens',
                    'es': gan_es,
                    'gt': gan_gt
                })

            # if len(probe_enc_sup_es) > 0:
            #     probe_enc_sup_es = self._unpad_and_concat(
            #         probe_enc_sup_es,
            #         alis_lens
            #     )
            #     scores_to_compute.append({
            #         'prefix': 'probe_tokens',
            #         'es': probe_enc_sup_es,
            #         'gt': alis_gt
            #     })
                # if self.pad_symbol is not None:
                #     not_pad = (alis_gt != self.pad_symbol)
                #     scores_to_compute.append({
                #         'prefix': 'probe_tokens_nonpad',
                #         'es': probe_enc_sup_es[not_pad],
                #         'gt': alis_gt[not_pad]
                #     })

            for stc in scores_to_compute:
                prefix = stc['prefix']
                es = stc['es']

                if alis_gt is not None:
                    gt = stc['gt']

                    all_scores[f'acc/{prefix}'] = (es == gt).mean()

                    # mapping_scores, mapping = self._mapping_metrics(
                    #     gt,
                    #     es,
                    #     prefix=prefix
                    # )
                    # all_scores.update(mapping_scores)

                    # Run the segmentation plottin with mapping
                    # if logger.is_currently_logging():
                    #     _, _, tokens = self.minibatch_loss_and_tokens(
                    #         first_batch
                    #     )
                    #     self.plot_input_and_alignments(
                    #         first_batch['features'],
                    #         alignment_es=tokens,
                    #         alignment_gt=first_batch['alignment'],
                    #         mapping=mapping,
                    #         imshow_kwargs=dict(cmap='Greys'),
                    #         log_suffix=f'{prefix}'
                    #     )

                    # clustering_scores = self._clustering_metrics(
                    #     gt,
                    #     es,
                    #     prefix=prefix
                    # )
                    # all_scores.update(clustering_scores)

                # perplexity_scores = self._perplexity_metrics(es, prefix=prefix)
                # all_scores.update(perplexity_scores)

        return all_scores

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

        gan_batch = self.gan_data_manipulator.prepare_gan_batch(
            encoder_output.data,
            batch,
            auto_length=True,
        )
        gen_output: torch.Tensor = self.gan_generator(gan_batch.data)
        tokens = gen_output.argmax(dim=-1)
        tokens_aligned = self.gan_data_manipulator.align_gan_output(
            tokens,
            gan_batch
        ).unsqueeze(dim=2).unsqueeze(dim=3)
        return (
            torch.tensor(0., requires_grad=True, device=feats.device),
                # rec_loss,
            {
                'gan_tokens': tokens,
                'gan_batch': gan_batch
            },  # details,
            tokens_aligned
        )
