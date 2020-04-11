import logging

import numpy as np
import torch

from distsup.modules.gan.data_types import GanConfig
from distsup.utils import (
    rleEncode,
    safe_squeeze, )


# DICT_SIZE = 70


class GanConcatedWindowsDataManipulation:
    def __init__(
        self,
        gan_config: GanConfig,
        encoder_length_reduction,
    ):
        self.encoder_length_reduction = encoder_length_reduction
        self.windows_size = gan_config.concat_window
        self.max_sentence_length = gan_config.max_sentence_length
        self.repeat = gan_config.repeat
        self.dictionary_size = gan_config.dictionary_size



    def generate_indexer(self, phrase_length) -> torch.tensor:
        concat_window_indexes = (
            (np.arange(self.windows_size) - self.windows_size // 2)[None, :]
            + (np.arange(phrase_length)[:, None])
        )
        concat_window_indexes[concat_window_indexes < 0] = 0
        concat_window_indexes[
            concat_window_indexes >= phrase_length
        ] = phrase_length - 1
        return concat_window_indexes

    def extract_alignment_data(self, alignment, length=None):
        if length is None:
            length=self.max_sentence_length
        train_bnd = torch.zeros(
            (alignment.shape[0], length),
            dtype=torch.long
        )
        train_bnd_range = train_bnd.clone()
        target = train_bnd.clone()
        lens = torch.zeros((alignment.shape[0]), dtype=torch.long)
        for i, algn in enumerate(alignment):
            rle, values = rleEncode(algn)
            _len = values.shape[0]
            if _len > length:
                logging.warning(
                    f'rle len [{_len}] exceeded max_sentence_length '
                    f'[{length}]'
                )
                _len = length
                rle = rle[:_len]
                values = values[:_len]
            lens[i] = _len
            train_bnd[i, :_len] = rle[:, 0]
            train_bnd_range[i, :_len] = rle[:, 1] - rle[:, 0]
            target[i, :_len] = values
        return train_bnd, train_bnd_range, target, lens

    def prepare_gan_batch(self, x, alignment, length=None):
        if length is None:
            length = self.max_sentence_length
        x = safe_squeeze(x, dim=2)
        batch_size, phrase_length, data_size = x.shape

        indexer = self.generate_indexer(phrase_length)
        windowed_x = x[:, indexer].view(
            (batch_size, phrase_length, self.windows_size * data_size)
        )

        expanded_x = torch.repeat_interleave(
            windowed_x,
            self.encoder_length_reduction,
            dim=1
        )

        train_bnd, train_bnd_range, target, lens = self.extract_alignment_data(
            alignment,
            length=length
        )

        random_pick = torch.clamp(
            (torch.randn(
                batch_size * self.repeat,
                length
            ) * 0.2 + 0.5),
            min=0.0,
            max=1.0,
        )

        sample_frame_ids = (
            train_bnd.repeat(self.repeat, 1).float()
            + random_pick * train_bnd_range.repeat(self.repeat, 1).float()
        ).long()

        batched_sample_frame = expanded_x.repeat(self.repeat, 1, 1)[
            np.arange(batch_size * self.repeat).reshape([-1, 1]),
            sample_frame_ids
        ]

        mask = (
            torch.arange(length)[None, :] < lens[:, None]
        ).repeat(self.repeat, 1)
        batched_sample_frame[~mask] = torch.eye(
            data_size,
            device=batched_sample_frame.device
        )[0].repeat(self.windows_size)

        return batched_sample_frame, target, lens
