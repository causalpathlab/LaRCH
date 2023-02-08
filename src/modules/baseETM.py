import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Multinomial, Dirichlet
from torch.distributions import kl_divergence as kl

import scvi
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data

class BaseETMEncoder(torch.nn.Module):
    def __init__(
        self,
        n_input,
        n_latent,
        n_hidden = 128,
        transform_exp = False
    ):
        """
        Encodes data of n_input dimensions into space of n_latent dimensions. Uses one layer fully-connected neural network

        Parameters
        ----------
        n_input
            Number of dimensions of the input

        n_hidden
            Number of hidden layers in the NN

        n_latent
            Number of dimensions of the output (latent factors)

        transform_exp
            Whether to perform exponential transformation

        """
        super().__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_latent)
        )

        self.transformation = None
        if transform_exp:
            self.transformation = torch.exp

    def forward(self, x: torch.Tensor):
        output = self.neural_net(x)
        if self.transformation:
            output = self.transformation(output)
        return output

class BaseETMDecoder(torch.nn.Module):
    def __init__(
        self,
        n_latent,
        n_output
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_output = n_output

        self.beta_mean = nn.Parameter(torch.randn(self.n_latent, self.n_output))
        self.beta_var = nn.Parameter(torch.randn(self.n_latent, self.n_output))

        self.theta_logsftm = nn.LogSoftmax(dim = -1)
        self.beta_logsftm = nn.LogSoftmax(dim = -1)
        self.rho_logsftm = nn.LogSoftmax(dim = -1)

    def forward(self, z: torch.Tensor):
        log_theta = self.theta_logsftm(z)

        log_beta = self.beta_logsftm(self.beta)

        log_rho = self.rho_logsftm(torch.mm(torch.exp(log_theta), torch.exp(log_beta)))

        return log_beta, log_theta, log_rho

class BaseETMModule(BaseModuleClass):

    def __init__(
        self,
        n_input,
        n_latent
    ):

        super().__init__()
        self.decoder = BaseETMDecoder(
            n_latent,
            n_input
        )

        self.mean_encoder = BaseETMEncoder(
            n_input,
            n_latent = n_latent
        )
        self.var_encoder = BaseETMEncoder(
            n_input,
            n_latent = n_latent,
            transform_exp = True
        )

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]

        return dict(x = x)

    @auto_move_data
    def inference(self, x):
        x_ = torch.log1p(x)

        qz_m = self.mean_encoder(x_)
        qz_v = self.var_encoder(x_)

        z = Normal(qz_m, torch.sqrt(qz_v)).rsample()

        return dict(qz_m = qz_m, qz_v = qz_v, z = z)

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        x = tensors[REGISTRY_KEYS.X_KEY]

        library = torch.sum(x, dim=1, keepdim=True)

        return dict(z = z, library = library)

    @auto_move_data
    def generative(self, z, library):

        log_beta, log_theta, log_rho = self.decoder(z)

        return dict(log_beta = log_beta, log_theta = log_theta, log_rho = log_rho)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]

        rho = torch.exp(generative_outputs["log_rho"])

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        # This might not work, Multinomial has default `total_count=1` which seems to cause problems
        # But according to the documentation, this is supposed to work
        log_lik = Multinomial(total_count = int(torch.max(library)), logits = rho).log_prob(x)

        prior_dist = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        var_post_dist = Normal(qz_m, torch.sqrt(qz_v))
        kl_divergence = kl(var_post_dist, prior_dist).sum(dim = 1)

        # This doesn't work, different dimensions
        elbo = log_lik - kl_divergence
        loss = torch.mean(-elbo)

        return LossRecorder(loss, -log_lik, kl_divergence, 0.0)
