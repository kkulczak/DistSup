from dataclasses import dataclass


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