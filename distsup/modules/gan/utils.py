import numpy as np
import torch
from torch import (
    autograd,
    nn,
)
from torch.autograd import Variable

from distsup import utils


class LReluCustom(nn.Module):
    def __init__(self, leak=0.1):
        super(LReluCustom, self).__init__()
        self.leak = leak

    def forward(self, x):
        return torch.max(x, self.leak * x)

def lrelu(x, leak=0.1):
    return torch.max(x, leak * x)

def softmax_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
    eps: float = 1e-20
):
    uniform = torch.rand(logits.shape, device=logits.device)
    noise = -torch.log(-torch.log(uniform + eps) + eps)
    y = logits + noise
    return nn.functional.softmax(y / temperature, dim=-1)


def compute_gradient_penalty(
    discriminator: torch.nn.Module,
    real_data: torch.tensor,
    generated_data: torch.tensor
):
    # ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master
    # /implementations/wgan_gp/wgan_gp.py
    '''

    :param input: state[index]
    :param network: actor or critic
    :return: gradient penalty
    '''
    batch_size = real_data.size()[0]
    device = generated_data.device
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data).to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated.requires_grad_(True)
    interpolated = Variable(
        interpolated,
        requires_grad=True
    ).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator.forward(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(
            prob_interpolated.size()
        ).to(device),
        create_graph=True,
        retain_graph=True
    )[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()


def assert_one_hot(x: torch.Tensor):
    x = x.view(-1, x.shape[-1]).cpu()
    ones = (x == 1.0).sum()
    zeros = (x == 0.0).sum()
    try:
        assert np.prod(x.shape) == (zeros + ones).item()
        assert (x.sum(dim=-1) == 1.0).all()
    except AssertionError as e:
        print(e)
        breakpoint()
        raise


def assert_as_target(x: torch.Tensor, target: torch.Tensor):
    x = x.cpu()
    target = target.cpu().long()
    if len(x.shape) == 4:
        x = utils.safe_squeeze(x, dim=2)
    if len(x.shape) == 3:
        es_tokens = x.argmax(dim=-1).long()
    else:
        es_tokens = x.long()
    assert (es_tokens == target).all().item() == 1

class AlignmentPrettyPrinter:
    def __init__(self, dataloader):
        try:
            alphabet = dataloader.dataset.alphabet.chars
        except AttributeError:
            alphabet = dataloader.dataset.dataset.alphabet.chars
        self.chars = {
            num: char
            for char, num in alphabet.items()
        }
        self.line_length = 120

    def show(self, x, target):
        x = x.view(-1)
        target = target.view(-1)
        assert len(target) == len(x)
        print('#' * self.line_length)
        for i in range(0, len(target), self.line_length):
            slc = slice(i, i + self.line_length)
            if i != 0:
                print('#' * self.line_length)
            print(''.join([self.chars[y.item()] for y in target[slc]]))
            print(''.join([self.chars[y.item()] for y in x[slc]]))
