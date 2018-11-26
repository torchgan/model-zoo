# Torch and Torchvision imports
import argparse
import torch
import torchvision
from torch.optim import Adam
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Torchgan imports
from torchgan import *
from torchgan.models import DCGANGenerator, DCGANDiscriminator
from torchgan.losses import GeneratorLoss, MinimaxDiscriminatorLoss
from torchgan.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", help="Choose whether to train on MNIST or CIFAR10, defaults to MNIST", default="mnist")
args = parser.parse_args()

def mnist_dataloader():
    train_dataset = dsets.MNIST(root='./mnist', train=True,
                                transform=transforms.Compose([transforms.Pad((2, 2)),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]),
                                download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    return train_loader

def cifar10_dataloader():
    train_dataset = dsets.CIFAR10(root='./cifar10', train=True,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]),
                                  download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    return train_loader

class BoundarySeekingGeneratorLoss(GeneratorLoss):
    def __init__(self, *args):
        super(BoundarySeekingGeneratorLoss, self).__init__(*args)
    def forward(self, dgz):
        dgz = torch.sigmoid(dgz)
        return 0.5 * torch.mean((torch.log(dgz) - torch.log(1 - dgz))**2)

network_params = {
        "generator": {"name": DCGANGenerator, "args": {"out_channels": 3 if args.dataset == "cifar10" else 1, "step_channels": 16}},
        "discriminator": {"name": DCGANDiscriminator, "args": {"in_channels": 3 if args.dataset == "cifar10" else 1, "step_channels": 16}}
}
optim_params = {
        "optimizer_generator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
        "optimizer_discriminator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},

}
losses = [BoundarySeekingGeneratorLoss(), MinimaxDiscriminatorLoss()]
trainer = Trainer(network_params, optim_params, losses, sample_size=64, epochs=20)

if args.dataset == "mnist":
    trainer(mnist_dataloader())
else:
    trainer(cifar10_dataloader())
