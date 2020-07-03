from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch


@dataclass
class GanConfig:
    concat_window: int
    dictionary_size: int
    max_sentence_length: int
    repeat: int
    gradient_penalty_ratio: float
    # Discriminator
    dis_steps: int
    dis_emb_size: int
    dis_hidden_1_size: int
    dis_hidden_2_size: int
    dis_learning_rate: float
    # Generator
    gen_hidden_size: int
    gen_steps: int
    gen_learning_rate: float
    # Optional
    supervised: bool = False
    use_all_letters: bool = False
    batch_inject_noise: float = 0.0
    dis_maxpool_reduction: int = 1
    filter_blanks: bool = False
    sample_from_middle_of_frame: bool = False
    backprop_ecoder: bool = False


@dataclass
class EncoderOutput:
    data: torch.Tensor
    lens: torch.Tensor


@dataclass
class GanAlignment:
    target: torch.Tensor
    lens: torch.Tensor
    train_bnd: torch.Tensor
    train_bnd_range: torch.Tensor


@dataclass
class GanBatch(GanAlignment):
    data: torch.Tensor
    batch: Dict[str, torch.Tensor]
