import numpy as np
import torch
from torch.nn.functional import pad

from distsup.utils import rleEncode

WINDOWS_SIZE = 5
REPEAT = 6
DICT_SIZE = 70
MAX_SENTENCE_LENGTH = 37


def generate_indexer(window_size, phrase_length) -> torch.tensor:
    concat_window_indexes = (
        (np.arange(window_size) - window_size // 2)[None, :]
        + (np.arange(phrase_length)[:, None])
    )
    concat_window_indexes[concat_window_indexes < 0] = 0
    concat_window_indexes[
        concat_window_indexes >= phrase_length
    ] = phrase_length - 1
    return concat_window_indexes


def extract_alignment_data(alignment):
    encoded = [rleEncode(x) for x in alignment]
    ranges = [x[0][:, 0].squeeze() for x in encoded]
    values = [x[1] for x in encoded]
    lens = torch.tensor([len(x) for x in values])

    # for x in encoded:
    #     print((x[0]), '\n\n\n',(x[1]))
    #     print((x).shape)
    # print(encoded)
    def padded_tensor(xs):
        return torch.stack([
            torch.cat([
                x,
                torch.zeros(
                    MAX_SENTENCE_LENGTH - x.shape[0],
                    dtype=torch.int
                )
            ])
            for x in xs
        ])

    train_bnd = padded_tensor(ranges)
    train_bnd_range = pad(
        train_bnd[:, 1:] - train_bnd[:, :-1],
        (0, 1, 0, 0),
    )
    train_bnd_range[train_bnd_range < 0] = 0.

    target = padded_tensor(values)
    return train_bnd, train_bnd_range, target, lens


def prepare_gan_batch(x, alignment, length_reduction=4):
    x = x.squeeze()
    batch_size, phrase_length, data_size = x.shape

    indexer = generate_indexer(WINDOWS_SIZE, phrase_length)
    windowed_x = x[:, indexer].view(
        (batch_size, phrase_length, WINDOWS_SIZE * data_size)
    )

    expanded_x = torch.repeat_interleave(
        windowed_x,
        length_reduction,
        dim=1
    )

    train_bnd, train_bnd_range, target, lens = extract_alignment_data(alignment)

    random_pick = torch.clamp(
        (torch.randn(batch_size * REPEAT, MAX_SENTENCE_LENGTH) + 0.5) * 0.2,
        min=0.0,
        max=1.0,
    )

    sample_frame_ids = (
        train_bnd.repeat(REPEAT, 1).type(torch.float)
        + random_pick * train_bnd_range.repeat(REPEAT, 1).type(torch.float)
    ).round().type(torch.long)
    print(sample_frame_ids[:2])

    batched_sample_frame = expanded_x.repeat(REPEAT, 1, 1)[
        np.arange(batch_size * REPEAT).reshape([-1, 1]),
        sample_frame_ids
    ]

    mask = (
        torch.arange(MAX_SENTENCE_LENGTH)[None, :] < lens[:, None]
    ).repeat(REPEAT, 1)
    batched_sample_frame[~mask] = 0.0

    return batched_sample_frame, target, lens


def prepare_windowed_example(x: torch.tensor, window_size=5):
    assert len(x.shape) == 2
    # indexer =

# if __name__ == '__main__':
#     x = torch.tensor([
#         [1, 2, 3],
#         [5, 5, 6]
#     ])
#     alignment = torch.tensor([
#         [0, 0, 0, 0, 1, 1, 1, 2, 2],
#         [1, 2, 2, 2, 2, 3, 3, 3, 4]
#
#     ])
#     temp = prepare_gan_batch(x, alignment, length_reduction=3)
