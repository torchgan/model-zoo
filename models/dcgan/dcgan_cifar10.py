# Torch and Torchvision imports
import torch
import torchvision
from torch.optim import Adam
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Torchgan imports
from torchgan import *
from torchgan.models import DCGANGenerator, DCGANDiscriminator
from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss,\
                            LeastSquaresDiscriminatorLoss, LeastSquaresGeneratorLoss
from torchgan.trainer import Trainer

# Change the root to your desired path
# NOTE: Once we have an integrated Datasets module in torchgan even this becomes obsolete
def cifar10_dataloader():
    train_dataset = dsets.CIFAR10(root='/data/avikpal', train=True,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                              transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]),
                                  download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    return train_loader

# Create an instance of the Trainer class with the parameter needed
# The models and images will be stored in `model` directory and `images` directory
trainer = Trainer({"generator": {"name": DCGANGenerator, "args": {"out_channels": 3, "step_channels": 16}},
                   "discriminator": {"name": DCGANDiscriminator, "args": {"in_channels": 3, "step_channels": 16}}},
                  {"optimizer_generator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}},
                   "optimizer_discriminator": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}},
                  [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()],
                  sample_size=64, epochs=20)

# Call the trainer with the data loader
trainer(cifar10_dataloader())

# Launch tensorboard to see the generated images and the loss
