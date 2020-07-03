# -*- coding: utf8 -*-
#   Copyright 2019 JSALT2019 Distant Supervision Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import copy
import io
import logging
import os
import pickle
import sys
import re
from typing import Optional
import zipfile

import torch.utils.data
import torch.nn.functional as F

# Future FIXME: extract Path from Aligner into its own class and just import
#  class Path
from distsup import utils
from distsup.modules.gan.utils import EncoderTokensProtos

sys.path.append('/home/kku/Documents/DistSup')
from distsup.alphabet import Alphabet


class TextScribbleLensDataset(torch.utils.data.Dataset):
    tokens_protos: Optional[EncoderTokensProtos] = None

    def __init__(
        self,
        mode='id',
        shape_as_image=False,
        eval_size_only=False,
        texts_path='data/texts_train.pickle',
        vocabulary='egs/scribblelens/tasman.alphabet.plus.space.mode5.json',
        tokens_protos=None,
        max_lenght=None,
        unmatched_real_sample=None,
    ):
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)
        self.alphabet = Alphabet(vocabulary)
        self.alignments = [
            torch.tensor(self.alphabet.symList2idxList(t))
            for t in self.texts
        ]
        if unmatched_real_sample is not None:
            if unmatched_real_sample:
                self.alignments = self.alignments[::2]
            else:
                self.alignments = self.alignments[1::2]

        self.alignments_lens = [None for _ in range(len(self.alignments))]
        if max_lenght is not None:
            for i in range(len(self.alignments)):
                length = self.alignments[i].shape[0]
                if length < max_lenght:
                    self.alignments[i] = torch.cat([
                        self.alignments[i],
                        torch.zeros(max_lenght - length, dtype=torch.long)
                    ])
                    self.alignments_lens[i] = length
                else:
                    self.alignments[i] = self.alignments[i][:max_lenght]
                    self.alignments_lens[i] = max_lenght

        if mode == 'id':
            self.features = [
                F.one_hot(a, num_classes=len(self.alphabet)).float()
                for a in self.alignments
            ]
        else:
            raise NotImplemented(mode)

        self.add_channels_dim = shape_as_image
        self.eval_size_only = eval_size_only
        self.metadata = {
            'alignment': {
                'type': 'categorical',
                'num_categories': len(self.alphabet)
            },
        }
        if tokens_protos is not None:
            self.tokens_protos = utils.construct_from_kwargs(
                tokens_protos
            )
            self.features = [
                utils.safe_squeeze(
                    self.tokens_protos.gen_sample(a[None, :]),
                    dim=0,
                )
                for a in self.alignments
            ]

    def __len__(self):
        if self.eval_size_only:
            return 512
        return len(self.alignments)

    def __getitem__(self, item):
        features = self.features[item]
        alignment = self.alignments[item]
        if self.add_channels_dim:
            features = features.unsqueeze(2)
        return {
            'alignment': alignment,
            'features': features,
            'alignment_len': self.alignments_lens[item],
            'features_len': self.alignments_lens[item],
        }
