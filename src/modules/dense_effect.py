import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal, Dirichlet,
from torch.distributions import kl_divergence as kl

import scvi
from scvi.nn import FCLayers
from scvi import _CONSTANTS, REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data

class DenseEncoder(nn.Module):
    def __init__(
        self,
        input_dims: int,
        n_layers: int,
        hidden_layer_dims: int,
        latent_dims: int,
        log_variational: bool = True
    ):

        super().__init__()
        self.log_variational = log_variational

        self.encoder = FCLayers(
            n_in = input_dims,
            n_layers = n_layers,
            n_out = hidden_layer_dims,
        )

        self.mean_encoder = nn.Linear(hidden_layer_dims, latent_dims)
        self.var_encoder = nn.Linear(hidden_layer_dims, latent_dims)

    def forward(self, x: torch.Tensor):
        if self.log_variational:
            x_ = torch.log1p(x)

        q = self.encoder(x_)
        q_m = self.mean_enocder(q)
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -5.0, 5.0))

        latent = Normal(q_m, torch.sqrt(q_v)).rsample()

        return q_m, q_v, latent

class DenseDecoder(nn.Module):
        def __init__(
            self,
            latent_dims,
            out_dims
        ):
            super().__init__()
            self.delta = nn.Parameter(torch.randn(1, out_dims))
            self.beta_mean = nn.Parameter(torch.randn(latent_dims, out_dims))
            self.beta_lnvar = nn.Parameter(torch.zeros(latent_dims, out_dims))

            self.logsftm == nn.LogSoftmax(dim=-1)

        def forward(self, z):
            theta = torch.exp(self.logsftm(z))

            z_beta = self.beta_reparam()
            beta = z_beta.add(self.beta_bias)

            return self.beta_mean, self.beta_lnvar, theta, beta

        def beta_reparam(self):
            lnv = torch.clamp(self.beta_lnvar, -5.0, 5.0)
            z_beta = Normal(self.beta_mean, torch.exp(self.beta_lnvar/2.))

            return z_beta

class DenseEffectModule(BaseModuleClass):
    def __init__(
        self,
        input_dims: int,
        latent_dims: int = 10,
        n_layers_encoder: int = 2,
        hidden_dims_encoder: int = 32,
        n_layers_decoder: int = 1,
        hidden_dims_decoder: int = 32,
        log_variational: bool = True
    ):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.n_layers_encoder = n_layers_encoder
        self.hidden_dims_encoder = hidden_dims_encoder
        self.n_layers_decoder = n_layers_decoder
        self.hidden_dims_decoder = hidden_dims_decoder
        self.log_variational = log_variational

        self.z_encoder = DenseEncoder(
            input_dims = input_dims,
            n_layers = n_layers_encoder,
            hidden_layer_dims = hidden_dims_encoder,
            latent_dims = latent_dims,
            log_variational = log_variational
        )

        self.decoder = DenseDecoder(
            latent_dims,
            input_dims
        )

    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY])

    @auto_move_data
    def inference(self, x: torch.Tensor) -> dict:
        x_ = x

        qz_m, qz_v, z = self.z_encoder(x_)

        return dict(qz_m = qz_m, qz_v = qz_v, z = z)

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]

        x = tensors[_CONSTANTS.X_KEY]
        library_size = torch.sum(x, dim=1, keepdim=True)

        return dict(z = z, library_size = library_size)

    @auto_move_data
    def generative(self, z: torch.Tensor, library) -> dict:

        beta_mean, beta_lnvar, theta, beta = self.decoder(z)

        return dict(beta_mean = beta_mean, beta_lnvar = beta_lnvar, theta = theta, beta = beta)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs
    ):
        x = tensors[_CONSTANTS.X_KEY]
        y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
