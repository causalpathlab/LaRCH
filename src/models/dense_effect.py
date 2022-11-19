import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.funcitonal as F
import torch.distributions
import torchvision
import matplotlib.pyplot as plt
import logging

# Dense effect model

class DenseEffectEncoder(nn.Module):
    def __init__(self, input_dims, latent_dims):
        super(DenseEffectEncoder, self).__init__()
        # define layers


        self.mu = nn.Linear(latent_dims, latent_dims)
        self.var = nn.Linear(latent_dims, latent_dims)

    def forward(self, X):
        # transform X
        x = torch.log1p(X)

        # feed into nn
        result = self.fc(x)


        # split result into mu and var
        mu = self.mu(result)
        log_var = self.var(result)

        z = self.reparameterize(mu, log_var)

        return z,mu,log_var

class DenseEffectDecoder(nn.Module):
    def __init__(self, latent_dims, out_dims):
        super(DenseEffectDecoder, self).__init__()

class DenseEffectModel(nn.Module):
    def __init__(self, input_dims, latent_dims):
        self.encoder = DenseEffectEncoder(input_dims, latent_dims)
        self.decoder = DenseEffectDecoder(latent_dims, input_dims)
