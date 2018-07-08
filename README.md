# IWAE - Tensorflow Implementation of Importance Weighted Autoencoder.
Variational Autoencoders are a class of generative modelling techniques that aim to learn a good representation of the distribution of the observed data x by imposing an underlying latent space z. By accurately approximating the posterior p(z|x)  it aims to maximize the lower bound to p(x).

However, VAEs make assumption that the posterior is approximately factorial which limits the capacity of the model. IWAEs use multiple samples from the approximate posterior to get a tighter lower bound on p(x), which allows you to fit posteriors that can potentially overcome the assumptions VAE lays out.

IWAE has been introduced and explained in the paper [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519) by Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov.

The experiments and results from this work are summarized in this poster (PDF version [here](https://github.com/neha191091/IWAE/blob/master/iwae/IWAE_Poster.pdf))

![alt text](https://github.com/neha191091/IWAE/blob/master/iwae/IWAE_Poster.jpg)

## Prerequisites for running the code
### Dataset: 
Load the required datasets by running **datasets_download.py**
### Python packages: 
tensorflow, tensorflow.python.debug, numpy, progressbar, sys, os .. etc.

## Running the experiments
This code allows you to train, evaluate and compare VAE and IWAE architectures on the mnist dataset. Additionally you can also plot samples from the posterior from the model. All this can be done by changing the parameters of the main code in **experiments.py** and running it. 
