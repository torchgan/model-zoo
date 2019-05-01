import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim import Adam
from torchgan.losses import (WassersteinDiscriminatorLoss,
                             WassersteinGeneratorLoss,
                             WassersteinGradientPenalty)
from torchgan.models import DCGANDiscriminator, DCGANGenerator
from torchgan.trainer import ParallelTrainer, Trainer


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
        # While backpropagating we only consider the output of the sigmoid
        # function
        x, = self.saved_tensors
        return grad_output * x * (1.0 - x), None, None, None


class BinaryNeurons(nn.Module):
    r"""Implements the Binary Neurons as described in
    https://arxiv.org/pdf/1810.04714.pdf

    Args:
        mode (str, optional): 2 choices - 'd' for Deterministic Binary Neurons
            and 's' for Stochastic Binary Neurons. Any other choice will result
            in the preactivation output.
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--data_dir",
        help="directory where mnist will be downloaded/is available",
        default="./")
    parser.add_argument(
        "-sc",
        "--step_channels",
        help="step channels for the generator and discriminators",
        type=int,
        default=16)
    parser.add_argument(
        "-t",
        "--threshold",
        help="The threshold for binarization",
        type=float,
        default=0.5)
    parser.add_argument(
        "--type",
        help="Type of Binarization to be used",
        choices=['s', 'd'],
        default='s')
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="The learning rate for the optimizers",
        type=float,
        default=0.0002)
    parser.add_argument(
        "--cpu",
        type=int,
        help="Set it to 1 if cpu is to be used for training",
        default=0)
    parser.add_argument(
        "-m",
        "--multigpu",
        choices=[0, 1],
        type=int,
        help="Choose 1 if multiple GPUs are available for training",
        default=0)
    parser.add_argument(
        "-l",
        "--list_gpus",
        type=int,
        nargs='+',
        help="List of GPUs to be used for training. Used iff -m is set to 1",
        default=[0, 1])
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch Size for training",
        default=32)
    parser.add_argument(
        "-s",
        "--sample_size",
        type=int,
        help="Number of Images Generated per Epoch",
        default=64)
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Place to store the trained model",
        default="./gman_")
    parser.add_argument(
        "-r",
        "--reconstructions",
        help="Directory to store the generated images",
        default="./gman_images")
    parser.add_argument(
        "-e",
        "--epochs",
        help="Total epochs for which the model will be trained",
        default=20,
        type=int)
    args = parser.parse_args()

    network_config = {
        "generator": {
            "name": DCGANGenerator,
            "args": {
                "encoding_dims":
                100,
                "out_channels":
                1,
                "step_channels":
                args.step_channels,
                "last_nonlinearity":
                BinaryNeurons(mode=args.type, threshold=args.threshold)
            },
            "optimizer": {
                "name": Adam,
                "args": {
                    "lr": args.learning_rate,
                    "betas": (0.5, 0.999)
                }
            }
        },
        "discriminator": {
            "name": DCGANDiscriminator,
            "args": {
                "in_channels": 1,
                "step_channels": args.step_channels
            },
            "optimizer": {
                "name": Adam,
                "args": {
                    "lr": args.learning_rate,
                    "betas": (0.5, 0.999)
                }
            }
        }
    }

    losses_list = [
        WassersteinGeneratorLoss(),
        WassersteinDiscriminatorLoss(),
        WassersteinGradientPenalty()
    ]

    if args.cpu == 0 and args.multigpu == 1:
        trainer = ParallelTrainer(
            network_config,
            losses_list,
            args.list_gpus,
            epochs=args.epochs,
            sample_size=args.sample_size,
            checkpoints=args.checkpoint,
            retain_checkpoints=1,
            recon=args.reconstructions,
        )
    else:
        if args.cpu == 1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0")
        trainer = Trainer(
            network_config,
            losses_list,
            device=device,
            epochs=args.epochs,
            sample_size=args.sample_size,
            checkpoints=args.checkpoint,
            retain_checkpoints=1,
            recon=args.reconstructions,
        )

    # Transforms to get Binarized MNIST
    dataset = dsets.MNIST(
        root=args.data_dir,
        train=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: x.convert('1')),
            transforms.ToTensor()
        ]),
        download=True)

    dataloader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    trainer(dataloader)

    trainer.complete()
