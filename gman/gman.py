import argparse

import torch
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim import Adam
from torchgan.losses import MinimaxDiscriminatorLoss, MinimaxGeneratorLoss
from torchgan.models import DCGANDiscriminator, DCGANGenerator
from torchgan.trainer import ParallelTrainer, Trainer


class MultiDiscriminatorMinimaxLoss(MinimaxDiscriminatorLoss):
    def __init__(self, *args, lambd=0.001, **kwargs):
        super(MultiDiscriminatorMinimaxLoss, self).__init__(*args, **kwargs)
        self.lambd = lambd

    def train_ops(
            self,
            generator,
            discriminator1,
            discriminator2,
            discriminator3,
            discriminator4,
            discriminator5,
            optimizer_discriminator1,
            optimizer_discriminator2,
            optimizer_discriminator3,
            optimizer_discriminator4,
            optimizer_discriminator5,
            real_inputs,
            device,
    ):
        batch_size = real_inputs.size(0)
        noise = torch.randn(batch_size, generator.encoding_dims, device=device)
        optimizer_discriminator1.zero_grad()
        optimizer_discriminator2.zero_grad()
        optimizer_discriminator3.zero_grad()
        optimizer_discriminator4.zero_grad()
        optimizer_discriminator5.zero_grad()
        fake = generator(noise).detach()
        dx1 = discriminator1(real_inputs)
        dx2 = discriminator2(real_inputs)
        dx3 = discriminator3(real_inputs)
        dx4 = discriminator4(real_inputs)
        dx5 = discriminator5(real_inputs)
        dgz1 = discriminator1(fake)
        dgz2 = discriminator2(fake)
        dgz3 = discriminator3(fake)
        dgz4 = discriminator4(fake)
        dgz5 = discriminator5(fake)
        V1 = self.forward(dx1, dgz1)
        V2 = self.forward(dx2, dgz2)
        V3 = self.forward(dx3, dgz3)
        V4 = self.forward(dx4, dgz4)
        V5 = self.forward(dx5, dgz5)
        exp_V1 = torch.exp(self.lambd * V1)
        exp_V2 = torch.exp(self.lambd * V2)
        exp_V3 = torch.exp(self.lambd * V3)
        exp_V4 = torch.exp(self.lambd * V4)
        exp_V5 = torch.exp(self.lambd * V5)
        loss = (exp_V1 * V1 + exp_V2 * V2 + exp_V3 * V3 + exp_V4 * V4 +
                exp_V5 * V5) / (exp_V1 + exp_V2 + exp_V3 + exp_V4 + exp_V5)
        loss.backward()
        optimizer_discriminator1.step()
        optimizer_discriminator2.step()
        optimizer_discriminator3.step()
        optimizer_discriminator4.step()
        optimizer_discriminator5.step()
        return loss.item()


class MultiDiscriminatorGeneratorLoss(MinimaxGeneratorLoss):
    def __init__(self, *args, lambd=0.001, **kwargs):
        super(MultiDiscriminatorGeneratorLoss, self).__init__(*args, **kwargs)
        self.lambd = lambd

    def train_ops(
            self,
            generator,
            discriminator1,
            discriminator2,
            discriminator3,
            discriminator4,
            discriminator5,
            optimizer_generator,
            batch_size,
            device,
    ):
        noise = torch.randn(batch_size, generator.encoding_dims, device=device)
        optimizer_generator.zero_grad()
        fake = generator(noise)
        dgz1 = discriminator1(fake)
        dgz2 = discriminator2(fake)
        dgz3 = discriminator3(fake)
        dgz4 = discriminator4(fake)
        dgz5 = discriminator5(fake)
        V1 = self.forward(dgz1)
        V2 = self.forward(dgz2)
        V3 = self.forward(dgz3)
        V4 = self.forward(dgz4)
        V5 = self.forward(dgz5)
        exp_V1 = torch.exp(self.lambd * V1)
        exp_V2 = torch.exp(self.lambd * V2)
        exp_V3 = torch.exp(self.lambd * V3)
        exp_V4 = torch.exp(self.lambd * V4)
        exp_V5 = torch.exp(self.lambd * V5)
        loss = (exp_V1 * V1 + exp_V2 * V2 + exp_V3 * V3 + exp_V4 * V4 +
                exp_V5 * V5) / (exp_V1 + exp_V2 + exp_V3 + exp_V4 + exp_V5)
        loss.backward()
        optimizer_generator.step()
        return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--data_dir",
        help="directory where mnist/cifar10 will be downloaded/is available",
        default="./")
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["mnist", "cifar10"],
        help="Dataset to use",
        default="mnist")
    parser.add_argument(
        "-sc",
        "--step_channels",
        help="step channels for the generator and discriminators",
        type=int,
        default=16)
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="The learning rate for the optimizers",
        type=float,
        default=0.0002)
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

    transformations = []
    if args.dataset == "cifar10":
        channels = 3
        dataset = dsets.CIFAR10
    else:
        channels = 1
        dataset = dsets.MNIST
        transformations.append(transforms.Resize((32, 32)))
    transformations.append(transforms.ToTensor())
    transformations.append(
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    network_configuration = {
        "generator": {
            "name": DCGANGenerator,
            "args": {
                "out_channels": channels,
                "step_channels": args.step_channels
            },
            "optimizer": {
                "name": Adam,
                "args": {
                    "lr": args.learning_rate,
                    "betas": (0.5, 0.999)
                }
            },
        },
        "discriminator1": {
            "name": DCGANDiscriminator,
            "args": {
                "in_channels": channels,
                "step_channels": args.step_channels
            },
            "optimizer": {
                "name": Adam,
                "args": {
                    "lr": args.learning_rate,
                    "betas": (0.5, 0.999)
                }
            },
        },
        "discriminator2": {
            "name": DCGANDiscriminator,
            "args": {
                "in_channels": channels,
                "step_channels": args.step_channels
            },
            "optimizer": {
                "name": Adam,
                "args": {
                    "lr": args.learning_rate,
                    "betas": (0.5, 0.999)
                }
            },
        },
        "discriminator3": {
            "name": DCGANDiscriminator,
            "args": {
                "in_channels": channels,
                "step_channels": args.step_channels
            },
            "optimizer": {
                "name": Adam,
                "args": {
                    "lr": args.learning_rate,
                    "betas": (0.5, 0.999)
                }
            },
        },
        "discriminator4": {
            "name": DCGANDiscriminator,
            "args": {
                "in_channels": channels,
                "step_channels": args.step_channels
            },
            "optimizer": {
                "name": Adam,
                "args": {
                    "lr": args.learning_rate,
                    "betas": (0.5, 0.999)
                }
            },
        },
        "discriminator5": {
            "name": DCGANDiscriminator,
            "args": {
                "in_channels": channels,
                "step_channels": args.step_channels
            },
            "optimizer": {
                "name": Adam,
                "args": {
                    "lr": args.learning_rate,
                    "betas": (0.5, 0.999)
                }
            },
        },
    }

    losses = [
        MultiDiscriminatorGeneratorLoss(),
        MultiDiscriminatorMinimaxLoss()
    ]

    if args.multigpu == 1:
        trainer = ParallelTrainer(
            network_configuration,
            losses,
            args.list_gpus,
            epochs=args.epochs,
            sample_size=args.sample_size,
            checkpoints=args.checkpoint,
            retain_checkpoints=1,
            recon=args.reconstructions,
        )
    else:
        trainer = Trainer(
            network_configuration,
            losses,
            epochs=args.epochs,
            sample_size=args.sample_size,
            checkpoints=args.checkpoint,
            retain_checkpoints=1,
            recon=args.reconstructions,
        )

    train_dataset = dataset(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transformations)

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    trainer(train_loader)

    trainer.complete()
