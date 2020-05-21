import numpy as np
import torch

from distsup.modules.gan.utils import EncoderTokensProtos


def test_gen_sample():
    B = 9999
    data = {
        'hidden': np.array([
            [B, 100, 101],
            [B, 100, 101],
            [B, 100, 102],
            [200, B, 201],
            [200, B, 202],
            [200, B, 203],
            [300, 301, B],
            [300, 302, B],
            [300, 303, B],
        ]),
        'gt': np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, ])
    }

    protos_class = EncoderTokensProtos(
        path=data,
        protos_per_token=3,
        num_tokens=3,
        preproc_softmax=False,
        deterministic=False
    )
    alignment = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 1, 0, 0]])
    sample = protos_class.gen_sample(alignment)
    for i, algn in enumerate(alignment.flatten()):
        assert (algn+1) * 100 in sample.view(-1, 3)[i]
