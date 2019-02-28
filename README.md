# TorchGAN: Model Zoo

Collection of Generative Adversarial Networks developed using TorchGAN

## Models Present

1. Generative MultiAdversarial Networks (GMAN)
   [Link to Paper](https://arxiv.org/pdf/1611.01673.pdf)
   Requires the `torchgan` master.

## Contribution Guidelines

We are open to accepting any model that you have built. The only things
to keep in mind are the following:

1. Keep the models simple and reuse features of torchgan if possible.
2. Have enough command line options for users to play with.
3. Once you are done run `isort` and `yapf` for formatting the code
   properly.

## FAQ

### How to run the Model?

To run these models you need to have `torchgan` installed.

Then simply move into the directory.

```bash
$ python3 <model name.py> --help
```

This will show you the configurable options that are available.

### Why do the models mostly use MNIST or CIFAR10?

The aim of this repository is to demonstrate the usage of torchgan. We believe
this is best done if users can simply download the script and run it without
having to download hundreds of GBs of dataset. However, we shall definitely
add more models in the future which are specifically designed for high resolution
data.

