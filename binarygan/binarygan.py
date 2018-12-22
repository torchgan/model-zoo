# Pytorch and Torchvision Imports
import torch
import torchvision
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# TorchGAN
import torchgan
from torchgan.models import DCGANGenerator, DCGANDiscriminator
from torchgan.trainer import Trainer
from torchgan.losses import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss,\
                            WassersteinGradientPenalty

# Transforms to get Binarized MNIST
dataset = dsets.MNIST(root='./mnist', train=True,
                      transform=transforms.Compose([transforms.Resize((32, 32)),
                                                    transforms.Lambda(lambda x: x.convert('1')),
                                                    transforms.ToTensor()]),
                      download=True)

dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

device = torch.device("cuda:0")

# Binary Neurons
class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mode, threshold=None, dist=None):
        x = torch.sigmoid(input)
        x_clone = x.clone()
        ctx.save_for_backward(x_clone)
        if mode == 'd':
            x[x >= threshold] = 1.0
            x[x < threshold] = 0.0
        elif mode == 's':
            z = dist.sample(x.size()).to(device)
            x[x >= z] = 1.0
            x[x < z] = 0.0
        return x

    @staticmethod
    def backward(self, grad_output):
        # While backpropagating we only consider the output of the sigmoid function
        x, = self.saved_tensors
        return grad_output * x * (1.0 - x), None, None, None

class BinaryNeurons(nn.Module):
    r"""Implements the Binary Neurons as described in https://arxiv.org/pdf/1810.04714.pdf

    Args:
        mode (str, optional): 2 choices - 'd' for Deterministic Binary Neurons and 's' for
            Stochastic Binary Neurons. Any other choice will result in the preactivation
            output.
        threshold (float, optional): The probability threshold.
    """
    def __init__(self, mode='d', threshold=0.5):
        super(BinaryNeurons, self).__init__()
        self.function = Binarize.apply
        self.mode = mode
        if self.mode == 'd':
            self.threshold = threshold
            self.dist = None
        elif self.mode == 's':
            self.dist = torch.distributions.uniform.Uniform(0.0, 1.0)
            self.threshold = None

    def forward(self, x):
        return self.function(x, self.mode, self.threshold, self.dist)


network_config = {
    "generator": {
        "name": DCGANGenerator,
        "args": {
            "encoding_dims": 100,
            "out_channels": 1,
            "step_channels": 32,
            "last_nonlinearity": BinaryNeurons()
        },
        "optimizer": {
            "name": Adam,
            "args": {
                "lr": 0.0001,
                "betas": (0.5, 0.999)
            }
        }
    },
    "discriminator": {
        "name": DCGANDiscriminator,
        "args": {
            "in_channels": 1,
            "step_channels": 32
        },
        "optimizer": {
            "name": Adam,
            "args": {
                "lr": 0.0001,
                "betas": (0.5, 0.999)
            }
        }
    }
}

losses_list = [WassersteinGeneratorLoss(), WassersteinDiscriminatorLoss(), WassersteinGradientPenalty()]

trainer = Trainer(network_config, losses_list, sample_size=64, epochs=1000,
                  recon="./images_binary", retain_checkpoints=1, device=device)

trainer(dataloader)

trainer.complete()
