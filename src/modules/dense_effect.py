import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal, Dirichlet,
from torch.distributions import kl_divergence as kl

import scvi
from scvi.nn import FCLayers
from scvi import REGISTRY_KEYS
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
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -4.0, 4.0))

        latent = Normal(q_m, q_v.sqrt()).rsample()

        return q_m, q_v, latent

# class DenseDecoder(nn.Module):
#     def __init__(
#         self,
#         input_dims,
#         output_dims
#     ):
#         super().__init__()
#         self.input_dims = input_dims
#         self.output_dims = output_dims
#         self.rho = nn.Parameter(torch.randn())

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

        self.decoder = DenseDecoder
