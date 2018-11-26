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
from torchgan.losses import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss
from torchgan.trainer import Trainer

# Change the root to your desired path
# NOTE: Once we have an integrated Datasets module in torchgan even this becomes obsolete
def mnist_dataloader():
    train_dataset = dsets.MNIST(root='/data/avikpal', train=True,
                                transform=transforms.Compose([transforms.Pad((2, 2)),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]),
                                download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    return train_loader

# Create an instance of the Trainer class with the parameter needed
# The models and images will be stored in `model` directory and `images` directory
trainer = Trainer(DCGANGenerator(out_channels=1, step_channels=16),
                  DCGANDiscriminator(in_channels=1, step_channels=16),
                  Adam, Adam, [WassersteinGeneratorLoss(), WassersteinDiscriminatorLoss()],
                  sample_size=64, epochs=20,
                  optimizer_generator_options={"lr": 0.0002, "betas": (0.5, 0.999)},
                  optimizer_discriminator_options={"lr": 0.0002, "betas": (0.5, 0.999)})

# Call the trainer with the data loader
trainer(mnist_dataloader())
