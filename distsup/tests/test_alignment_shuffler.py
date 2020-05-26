import os
from unittest import TestCase

import torch
import numpy as np

from distsup.modules.gan.alignment_shuffler import AlignmentShuffler


class TestAlignmentShuffler(TestCase):
    # def test_fix_empty_data(self):
    #     pass
    #
    def test_mode_protos(self):
        als = AlignmentShuffler(
            mode='protos',
            path='../../data/encoder_letter_prototypes.npz'
        )
        x = torch.zeros(size=(32, 250), dtype=torch.long)
        res = als.apply_noise(x)
        self.assertGreaterEqual(
            (res == 0).float().mean(),
            0.85
        )


    def test_mode_id(self):
        als = AlignmentShuffler(
            mode='id'
        )
        x = torch.randint(als.dict_size, size=(32, 250))
        res = als.apply_noise(x)
        self.assertEqual((x == res).float().mean().item(), 1)

    def test_mode_constant_nosie_as_id(self):
        als = AlignmentShuffler(
            mode='constant',
            constant_noise=0.0,
        )
        x = torch.randint(als.dict_size, size=(32, 250))
        res = als.apply_noise(x)
        self.assertEqual((x == res).float().mean().item(), 1)

    def test_mode_constant_nosie_max(self):
        als = AlignmentShuffler(
            mode='constant',
            constant_noise=1.0,
        )
        x = torch.randint(als.dict_size, size=(32, 250))
        res = als.apply_noise(x)
        self.assertEqual((x == res).float().mean().item(), 0)



