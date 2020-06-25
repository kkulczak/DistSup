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
import zipfile

import torch.utils.data
import torch.nn.functional as F
import PIL.Image
import torchvision
import numpy as np
import pandas as pd

# Future FIXME: extract Path from Aligner into its own class and just import
#  class Path
import distsup
from distsup import aligner

import egs.scribblelens.utils
from distsup.alphabet import Alphabet
from distsup.utils import construct_from_kwargs


class TextScribbleLensDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mode='id',
        texts_path='data/texts_train.pickle',
        vocabulary='egs/scribblelens/tasman.alphabet.plus.space.mode5.json'
    ):
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)
        self.alphabet = Alphabet(vocabulary)
        self.alignments = [
            torch.tensor(self.alphabet.symList2idxList(t))
            for t in self.texts
        ]
        if mode == 'id':
            self.features = [
                F.one_hot(a, num_classes=len(self.alphabet)).float()
                for a in self.alignments
            ]
        else:
            raise NotImplemented(mode)

        self.metadata = {
            'alignment': {
                'type': 'categorical',
                'num_categories': len(self.alphabet)
            },
            # 'text': {
            #     'type': 'categorical',
            #     'num_categories': len(self.alphabet)
            # },
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            'alignment': self.alignments[item],
            'features': self.features[item],
        }
