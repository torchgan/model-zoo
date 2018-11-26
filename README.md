# TorchGAN Model Zoo

Examples of Generative Adversarial Networks built using torchgan.

This repository is meant to be used to understand the functionality of
torchgan. Currently we don't provide any form of trained models here but
it might be added later on.

## Models Present

* DCGAN (Deep Convolutional Generative Adversarial Network)
* BEGAN (Boundary Equilibrium Generative Adversarial Network)
* EBGAN (Energy Based Generative Adversarial Network)
* WGAN (Wasserstein Generative Adversarial Network)
* WGAN with Gradient Penalty
* BGAN (Boundary Seeking Generative Adversarial Network)

## Contributions

Contributions are always welcome. Just open an issue before you start any
work, this allows us to prevent any form of redundant work.

Follow the style guidelines of `torchgan`, simply use `flake8` using the
config file present in `torchgan`.

Also the following structure must be followed:
1. Create a directory and place your model inside it.
2. If there are any external dependencies add a `requirements.txt` file.
3. A small `README` with the links to the research paper and 
   execution guidelines.
4. The script must accept command line arguments for training.
5. Add your model to the list in this README.

## Issues

Feel free to open issues for model requests and if you find any trouble
with the existing models present here.
