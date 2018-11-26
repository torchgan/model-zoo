# DCGAN

## Introduction

Contains minimal examples on how to use `torchgan`. DCGAN is one of the models that is supported
out of box by `torchgan`. In this case we use a custom version of DCGAN, which we prefer to call
`SmallDCGAN`. It is simply adapted to handle smaller images.

## Usage

It is recommended that you use the `jupyter notebook` corresponding to these models. But in case you prefer
to use this, you need to set a few of the parameters:

1. Set the `root` to where you want to store the dataset.
2. If you donot have a gpu, pass a parameter `device = torch.device("cpu")` to the Trainer.
3. You might want to comment out one of the two models, they simply demonstrate how we can seamlessly switch between losses.

## Results

### DCGAN_MNIST Least Squares Samples

![DCGAN_MNIST Least Squares Samples](./images/dcgan_mnist_ls.gif)

### DCGAN_MNIST Minimax Samples

![DCGAN_MNIST Minimax Samples](./images/dcgan_mnist_minimax.gif)

### DCGAN_CIFAR10 Minimax Samples

![DCGAN_CIFAR10 Minimax Samples](./images/dcgan_cifar10_minimax.gif)

## Contributors

1. Avik Pal [@avik-pal]
