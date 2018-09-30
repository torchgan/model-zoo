# TorchGAN model-zoo

Examples of Generative Adversarial Networks built using torchgan.

This repository is meant to be used to understand the functionality of
torchgan. Currently we donot provide any form of trained models here but
it might be added later on.

## Contributions

Contributions are welcome. Before starting to work on a model open an
issue for the same. This is just to make sure that you avoid redundant
work. Also be sure to follow the following guidelines :-

1. A Jupyter Notebook is needed for every model that is being added. The
   jupyter notebook must have a minimal working example using a simple
   dataset like `MNIST` or `CIFAR10`, however feel free to use more
   complicated ones. Add this to the `notebooks` directory. If the
   notebook demonstrates `DCGAN` on `MNIST` name place it in
   `dcgan/DCGAN_MNIST.ipynb`.
2. A python file for the model must be places in the `models` directory.
   Follow the same structure as the notebook.
3. Place a `README.md` file in both these directories. It must contain
   the `link` to the paper and a section `Contributors` where list out
   all the names of the people who have contributed in the form of
   `<FULL NAME>[<github handle>]`.

## Issues

If you want a new model to be added to the model-zoo open an issue with
the `tag`, `[NEW MODEL]`.<br>
If you are working on a model, open an issue with the tag `[WIP]`.<br>
For any usage related issues feel free to reach out on Slack. Also any
bugs related to the core library needs to be filed in `torchgan`
repository.
