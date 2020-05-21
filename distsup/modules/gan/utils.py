import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import (
    autograd,
    nn,
)
from torch.autograd import Variable
from torch.utils.data import Dataset

from distsup import utils
from distsup.data import ChunkedDataset, FixedDatasetLoader


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
    logging.warning('assert_as_target is turned off')
    return
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


class AlignmentDataset(Dataset):
    def __init__(self, alignments, target_height=32):
        super(AlignmentDataset).__init__()
        self.alignments = alignments
        self.target_height = target_height

    def __len__(self):
        return len(self.alignments)

    def __getitem__(self, item):
        algn = self.alignments[item]
        return {
            'alignment': algn
        }


def generate_alignemnt_dataset(vanilla_dataloader):
    alignments = [x['alignment'] for x in
        vanilla_dataloader.dataset.dataset.data]
    return FixedDatasetLoader(
        dataset=ChunkedDataset(
            dataset=AlignmentDataset(
                alignments,
                target_height=vanilla_dataloader.dataset.dataset.target_height
            ),
            chunk_len=vanilla_dataloader.dataset.chunk_len,
            varlen_fields=['alignment'],
            drop_fields=vanilla_dataloader.dataset.drop_fields,
            training=vanilla_dataloader.dataset.training,
            transform=vanilla_dataloader.dataset.transform,
            oversample=vanilla_dataloader.dataset.oversample,
            pad_with_zeros_if_short=vanilla_dataloader.dataset
                .pad_with_zeros_if_short,
        ),
        field_names=vanilla_dataloader.field_names,
        rename_dict=vanilla_dataloader.rename_dict,
        batch_size=vanilla_dataloader.batch_size,
        drop_last=vanilla_dataloader.drop_last,
        shuffle=True,
        num_workers=0,
    )


class EncoderTokensProtos:
    protos: torch.Tensor
    protos_per_token: int
    tokens_size: int

    def __init__(
        self,
        path,
        protos_per_token,
        num_tokens=68,
        deterministic=True,
        preproc_softmax=False,
    ) -> None:

        st0 = np.random.get_state()
        if deterministic:
            np.random.seed(0)

        arr = np.load(path)
        if preproc_softmax:
            data_hidden = F.softmax(
                torch.from_numpy(arr['hidden']),
                dim=1,
            ).numpy()
        else:
            data_hidden = arr['hidden']

        recognized = arr['gt'] == data_hidden.argmax(axis=1)
        hidden = data_hidden[recognized]
        gt = arr['gt'][recognized]
        unrecognized = data_hidden[~recognized]

        res = []
        for i in range(num_tokens):
            proto: np.ndarray = hidden[gt == i]
            if proto.size == 0:
                proto = unrecognized
            size = proto.shape[0]
            chosen = np.random.choice(size, size=protos_per_token)
            res.append(proto[chosen])

        self.protos_per_token = protos_per_token
        self.tokens_size = hidden.shape[1]
        self.protos = torch.from_numpy(np.stack(res, axis=0))

        if deterministic:
            np.random.set_state(st0)

    def gen_sample(self, alignment: torch.Tensor) -> torch.Tensor:
        batch_size, phrase_length = alignment.shape
        ids = torch.arange(batch_size * phrase_length) * self.protos_per_token
        rand_ids = torch.randint_like(ids, high=self.protos_per_token)
        sample = (
            self.protos[alignment.view(-1)]
                .view(-1, self.tokens_size)[ids + rand_ids]
                .view(batch_size, phrase_length, self.tokens_size)
        )

        return sample.to(device=alignment.device)
