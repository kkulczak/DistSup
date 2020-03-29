import numpy as np
import torch
from torch.nn.functional import pad

from distsup.utils import (
    rleEncode,
)


# DICT_SIZE = 70


class GanConcatedWindowsDataManipulation:
    def __init__(
        self,
        encoder_length_reduction,
        concat_window=5,
        max_sentence_length=37,
        repeat=6,
    ):
        self.encoder_length_reduction = encoder_length_reduction
        self.windows_size = concat_window
        self.max_sentence_length = max_sentence_length
        self.repeat = repeat
        pass

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

    def extract_alignment_data(self, alignment):
        encoded = [rleEncode(x) for x in alignment]
        ranges = [(x[0][:, 0]).squeeze() for x in encoded]
        values = [x[1] for x in encoded]
        lens = torch.tensor([len(x) for x in values])

        def padded_tensor(xs):
            xs = [
                x if len(x.shape) != 0  else torch.tensor([x])
                for x in xs
            ]
            return torch.stack([
                torch.cat([
                    x,
                    torch.zeros(
                        self.max_sentence_length - x.shape[0],
                        dtype=torch.int
                    )
                ])
                for x in xs
            ])

        train_bnd = padded_tensor(ranges)
        train_bnd_range = pad(
            train_bnd[:, 1:] - train_bnd[:, :-1],
            [0, 1, 0, 0],
        )
        train_bnd_range[train_bnd_range < 0] = 0.

        target = padded_tensor(values)
        return train_bnd, train_bnd_range, target, lens

    def prepare_gan_batch(self, x, alignment):
        x = x.squeeze()
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
            alignment)

        random_pick = torch.clamp(
            (torch.randn(batch_size * self.repeat,
                         self.max_sentence_length) + 0.5) * 0.2,
            min=0.0,
            max=1.0,
        )

        sample_frame_ids = (
            train_bnd.repeat(self.repeat, 1).type(torch.float)
            + random_pick * train_bnd_range.repeat(self.repeat, 1).type(
            torch.float)
        ).round().type(torch.long)

        batched_sample_frame = expanded_x.repeat(self.repeat, 1, 1)[
            np.arange(batch_size * self.repeat).reshape([-1, 1]),
            sample_frame_ids
        ]

        mask = (
            torch.arange(self.max_sentence_length)[None, :] < lens[:, None]
        ).repeat(self.repeat, 1)
        batched_sample_frame[~mask] = 0.0

        return batched_sample_frame, target, lens
