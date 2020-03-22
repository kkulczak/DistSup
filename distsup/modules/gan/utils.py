import torch
from torch import nn


class LReluCustom(nn.Module):
    def __init__(self, leak=0.1):
        super(LReluCustom, self).__init__()
        self.leak = leak

    def forward(self, x):
        return torch.max(x, self.leak * x)


def softmax_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
    eps: float = 1e-20
):
    uniform = torch.rand(logits.shape, device=logits.device)
    noise = -torch.log(-torch.log(uniform + eps) + eps)
    y = logits + noise
    return nn.functional.softmax(y / temperature, dim=-1)


def generate_inter_sample(
    fake: torch.Tensor,
    real: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        alpha = torch.rand([real.shape[0], 1, 1], device=fake.device)
        inter_sample = real + alpha * (fake - real)
        return inter_sample
