# vpal
This repository contains implementations and tests for the Variable Projected Augmented Lagrangian (VPAL) algorithm, along with supporting components and example experiments.

Contents

vpal
Core VPAL algorithm implementation. This is the main solver used to reconstruct signals/images under various forward operators and regularization.

testidentity
Example script where the forward operator A is the identity matrix. Useful for validating that the solver and regularization behave as expected without additional distortion.

testblur
Example script where the forward operator applies a blur. Demonstrates VPAL’s ability to deblur images under different noise and regularization settings.

D
Contains implementations of regularization operators (e.g., finite difference). These define the D in the VPAL formulation, controlling the type of prior enforced on the reconstruction.

autoencoder
Code for training an autoencoder on MNIST. The trained decoder can be used as part of the forward operator in VPAL’s latent space experiments.

mnist
Contains the .pth checkpoint file of the trained autoencoder.

test3
Test script for running VPAL in the autoencoder’s latent space, using the decoder as part of the forward model. Demonstrates reconstruction from blurred latent representations.

Usage

Each test script is self-contained and can be run independently after installing dependencies. Adjust hyperparameters such as mu, lambda_, and maxIter in the scripts to experiment with reconstruction quality and convergence.
