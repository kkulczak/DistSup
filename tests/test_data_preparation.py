from dataclasses import asdict

import torch

from distsup.modules.gan.data_preparation import align_gan_output
from distsup.modules.gan.data_types import GanBatch, GanConfig

gan_config = GanConfig(
    concat_window=1,
    repeat=1,
    dictionary_size=68,
    max_sentence_length=12,
    gradient_penalty_ratio=10.0,
    use_all_letters=False,
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

    new_alignment = align_gan_output(
        x,
        GanBatch(
            **asdict(gan_alignment),
            data=torch.tensor(0.),
            batch={'alignment': alignment},
        )
    )

    assert (new_alignment == alignment).all()
