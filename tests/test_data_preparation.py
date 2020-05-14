from copy import copy, deepcopy
from dataclasses import asdict
import os

import torch
import torch.nn.functional as F

from distsup.data import FixedDatasetLoader
from distsup.modules.gan.data_types import GanBatch, GanConfig

gan_config = GanConfig(
    concat_window=1,
    repeat=1,
    dictionary_size=68,
    max_sentence_length=12,
    gradient_penalty_ratio=10.0,
    use_all_letters=False,
    batch_inject_noise=0.0,
    dis_steps=3,
    dis_emb_size=256,
    dis_hidden_1_size=256,
    dis_hidden_2_size=256,
    dis_learning_rate=0.002,
    gen_steps=1,
    gen_hidden_size=128,
    gen_learning_rate=0.001,
)


def test_prepare_gan_batch():
    from distsup.modules.gan.data_preparation import \
        GanConcatedWindowsDataManipulation
    data_manipulator = GanConcatedWindowsDataManipulation(
        gan_config,
        encoder_length_reduction=4,
    )

    encoder_output = torch.arange(10, dtype=torch.float).unsqueeze(
        dim=0).unsqueeze(dim=2)
    alignment = torch.tensor([
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
    ]).unsqueeze(dim=0)
    expexted_output = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7,
    ], dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=2)
    for _ in range(10):
        gan_batch = data_manipulator.prepare_gan_batch(
            encoder_output,
            {'alignment': alignment},
            auto_length=False,
        )
        assert (gan_batch.data == expexted_output).all()


def test_align_gan_output():
    from distsup.modules.gan.data_preparation import \
        GanConcatedWindowsDataManipulation
    data_manipulator = GanConcatedWindowsDataManipulation(
        gan_config,
        encoder_length_reduction=4,
    )

    alignment = torch.tensor(
        [0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]

    ).unsqueeze(dim=0)
    gan_alignment = data_manipulator.extract_alignment_data(
        alignment,
        auto_length=True
    )
    x = torch.tensor([*list(range(9)), 0]).unsqueeze(dim=0)

    new_alignment = data_manipulator.align_gan_output(
        x,
        GanBatch(
            **asdict(gan_alignment),
            data=torch.tensor(0.),
            batch={'alignment': alignment},
        )
    )

    assert (new_alignment == alignment).all()


def test_prepare_dataset():
    print(os.listdir('../egs'))
    dataloader = FixedDatasetLoader(
        **{
            'batch_size': 1,
            'dataset': {
                'class_name': 'distsup.data.ChunkedDataset',
                'dataset': {
                    'class_name': 'egs.scribblelens.data.ScribbleLensDataset',
                    'root': '../data/scribblelens.corpus.v1.2.zip',
                    'split': 'train',
                    'alignment_root': '../data/scribblelens.paths.1.4b.zip',
                    'vocabulary':
                        '../egs/scribblelens/tasman.alphabet.plus.space'
                        '.mode5.json'
                },
                'chunk_len': 96,
                'training': True,
                'varlen_fields': ['image', 'alignment'],
                'drop_fields': ['text', 'alignment_rle', 'page_side',
                    'page']
            },
            'rename_dict': {'image': 'features'},
            'shuffle': False,
            'num_workers': 4,
            'drop_last': False
        }
    )
    global gan_config
    config = deepcopy(gan_config)
    config.max_sentence_length = 96
    from distsup.modules.gan.data_preparation import \
        GanConcatedWindowsDataManipulation
    data_manipulator = GanConcatedWindowsDataManipulation(
        gan_config=config,
        encoder_length_reduction=1
    )

    for batch in dataloader:
        encoder_output = F.one_hot(
            batch['alignment'].long(),
            gan_config.dictionary_size,
        ).float()
        gan_batch = data_manipulator.prepare_gan_batch(
            encoder_output,
            batch,
            auto_length=False,
            force_single_concat_window=True,
        )
        from distsup.modules.gan.utils import assert_one_hot,  assert_as_target
        assert_one_hot(gan_batch.data)
        assert_as_target(gan_batch.data, gan_batch.target)
        assert_as_target(
            data_manipulator.align_gan_output(
                gan_batch.data.argmax(dim=-1),
                gan_batch,
            ),
            gan_batch.batch['alignment'],
        )
        data_manipulator.align_gan_output(
            gan_batch.data.argmax(dim=-1),
            gan_batch,
        )


def test_inject_noise():
    from distsup.modules.gan.data_preparation import \
        GanConcatedWindowsDataManipulation
    config = copy(gan_config)
    config.batch_inject_noise = 0.5
    data_manipulator = GanConcatedWindowsDataManipulation(
        config,
        encoder_length_reduction=4,
    )
    x = torch.arange(5 * 100).view(5, 100, 1).repeat_interleave(5, dim=2)
    y = x.clone()

    data_manipulator.inject_noise(y)

    assert (0.4 < (y.sum(dim=2) == 1).float().mean().item() < 0.6)

    config.batch_inject_noise = 0.0
    data_manipulator = GanConcatedWindowsDataManipulation(
        config,
        encoder_length_reduction=4,
    )
    x = torch.arange(10, 5 * 100 + 10).view(5, 100, 1).repeat_interleave(5, dim=2)
    y = x.clone()

    data_manipulator.inject_noise(y)

    assert (y.sum(dim=2) != 1).all()

