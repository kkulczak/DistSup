from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator
import torch

from .sickit_extension import plot_confusion_matrix


class AlignmentShuffler:
    ground_truth: Optional[np.ndarray]
    protos: Optional
    cums_prob: np.ndarray

    def __init__(
        self,
        mode: str,
        path='',
        constant_noise=0.1,
        dict_size: int = 68,
    ):
        av_modes = ['id', 'constant', 'protos']
        if mode not in av_modes:
            raise AssertionError(f'mode should be one of: {av_modes}')
        self.dict_size = dict_size

        if mode == 'protos':
            self.cums_prob = self.probs_from_file(path)
        elif mode == 'id':
            self.cums_prob = np.eye(self.dict_size).cumsum(axis=1)
        elif mode == 'constant':
            positive_value = 1. - constant_noise
            probs = np.full(
                shape=(self.dict_size, self.dict_size),
                fill_value=constant_noise / (self.dict_size - 1),
            )
            probs[np.eye(self.dict_size, dtype=bool)] = positive_value
            self.cums_prob = probs.cumsum(axis=1)

    def probs_from_file(self, path):
        stored_values = np.load(path)
        self.ground_truth: np.ndarray = stored_values['gt']
        self.protos = stored_values['hidden']
        self.fix_empty_data()
        cm = metrics.confusion_matrix(
            self.ground_truth,
            self.protos.argmax(axis=1),
        )

        norm_cm = cm / cm.sum(
            axis=1,
            keepdims=True
        )

        return norm_cm.cumsum(axis=1)

    def fix_empty_data(self):
        all_tokens = np.ones(self.dict_size, dtype=int)
        all_tokens[np.unique(self.ground_truth)] = 0
        missing_tokens = np.nonzero(all_tokens)[0]

        self.ground_truth = np.concatenate([
            self.ground_truth,
            np.tile(missing_tokens, self.protos.shape[1]),
        ])
        self.protos = np.concatenate([
            self.protos,
            np.tile(
                np.eye(self.dict_size),
                missing_tokens.size
            ).reshape(-1, self.dict_size)
        ])

    def apply_noise(self, alignment: torch.Tensor) -> torch.Tensor:
        batch_size, phrase_length = alignment.shape
        alignment = alignment.cpu().numpy()
        propabilities: np.ndarray = self.cums_prob[alignment.flatten()]
        sample: np.ndarray = np.random.rand(alignment.size)[:, None]
        choices = (sample < propabilities).argmax(axis=1)
        return torch.from_numpy(choices.reshape(batch_size, phrase_length))
