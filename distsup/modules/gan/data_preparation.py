import dataclasses
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from distsup.modules.gan.data_types import GanAlignment, GanBatch, GanConfig
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
        self.config = gan_config
        self.encoder_length_reduction = encoder_length_reduction
        self.windows_size = gan_config.concat_window
        self.max_sentence_length = gan_config.max_sentence_length
        self.repeat = gan_config.repeat
        self.dictionary_size = gan_config.dictionary_size
        self.use_all_letters = gan_config.use_all_letters

    def to_cuda(self, batch: GanBatch) -> GanBatch:
        return GanBatch(
            target=batch.target.to('cuda'),
            lens=batch.lens.to('cuda'),
            train_bnd=batch.train_bnd.to('cuda'),
            train_bnd_range=batch.train_bnd_range.to('cuda'),
            data=batch.data.to('cuda'),
            batch=batch.batch
        )

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

    def extract_alignment_data(
        self,
        alignment: torch.Tensor,
        auto_length: bool = True,
    ) -> GanAlignment:
        if auto_length:
            length = 200
        else:
            length = self.max_sentence_length
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
                logging.error(
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
        if auto_length:
            longest_phrase = lens.max().item()
        else:
            longest_phrase = self.max_sentence_length
        return GanAlignment(
            train_bnd=train_bnd[:, :longest_phrase],
            train_bnd_range=train_bnd_range[:, :longest_phrase],
            target=target[:, :longest_phrase],
            lens=lens,
        )

    def prepare_gan_batch(
        self,
        x: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        auto_length: bool = True,
        force_single_concat_window=False,
    ) -> GanBatch:

        alignment = batch['alignment'].cpu()

        if len(x.shape) != 3:
            x = safe_squeeze(x, dim=2)
        batch_size, phrase_length, data_size = x.shape

        if force_single_concat_window:
            windowed_x = x
        else:
            indexer = self.generate_indexer(phrase_length)
            windowed_x = x[:, indexer].view(
                (batch_size, phrase_length, self.windows_size * data_size)
            )

        if self.use_all_letters:
            if self.config.batch_inject_noise != 0.0:
                self.inject_noise(windowed_x)
            return GanBatch(
                target=alignment.long(),
                lens=torch.full(
                    (batch_size,),
                    fill_value=windowed_x.shape[1],
                    dtype=torch.long
                ),
                train_bnd=torch.Tensor([0.]),
                train_bnd_range=torch.Tensor([0.]),
                data=windowed_x,
                batch=batch,
            )

        expanded_x = torch.repeat_interleave(
            windowed_x,
            self.encoder_length_reduction,
            dim=1
        )

        gan_alignment = self.extract_alignment_data(
            alignment,
            auto_length=auto_length,
        )

        length = gan_alignment.target.shape[1]
        random_pick = torch.clamp(
            (torch.randn(
                batch_size * self.repeat,
                length
            ) * 0.2 + 0.5),
            min=0.0,
            max=1.0,
        )

        sample_frame_ids = (
            gan_alignment.train_bnd.repeat(self.repeat, 1).float()
            + random_pick * gan_alignment.train_bnd_range.repeat(self.repeat,
                                                                 1).float()
        ).long()

        batched_sample_frame = expanded_x.repeat(self.repeat, 1, 1)[
            np.arange(batch_size * self.repeat).reshape([-1, 1]),
            sample_frame_ids
        ]

        if self.config.batch_inject_noise != 0.0:
            self.inject_noise(batched_sample_frame)

        mask = (
            torch.arange(length)[None, :] < gan_alignment.lens[:, None]
        ).repeat(self.repeat, 1)
        batched_sample_frame[~mask] = torch.eye(
            data_size,
            device=batched_sample_frame.device
        )[0].repeat(self.windows_size if not force_single_concat_window else 1)

        return GanBatch(
            **dataclasses.asdict(gan_alignment),
            data=batched_sample_frame,
            batch=batch,
        )

    def align_gan_output(
        self,
        x: torch.Tensor,
        batch: GanBatch
    ) -> torch.Tensor:
        if self.use_all_letters:
            return x
        output = torch.zeros_like(batch.batch['alignment'])
        for idx, (bnd, _range, _len) in enumerate(zip(
            batch.train_bnd,
            batch.train_bnd_range,
            batch.lens
        )):
            for i in range(_len.item()):
                output[idx, bnd[i]: bnd[i] + _range[i] + 1] = x[idx, i]
        return output

    def inject_noise(self, x: torch.Tensor) -> None:
        batch_size, phrase_length, data_size = x.shape
        noise = F.one_hot(
            torch.randint(
                data_size,
                size=(batch_size, phrase_length),
            ),
            num_classes=data_size,
        )
        inject_noise_ids = torch.rand((batch_size, phrase_length)) > (
            1.0 - self.config.batch_inject_noise)
        if self.config.batch_inject_noise == 0.0:
            assert inject_noise_ids.any().item() == 0

        x[inject_noise_ids] = noise[inject_noise_ids]
