import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Multinomial, Dirichlet
from torch.distributions import kl_divergence as kl

import scvi
from scvi.nn import FCLayers
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data

from baseETM import BaseETMModule, BaseETMEncoder, BaseETMDecoder

class SSLETMModule(BaseETMModule):
    def __init__(
        self,
        n_input,
        n_latent,
        # predetermined pi?
        pi
    ):
        super().__init__(
            n_input,
            n_latent)

        self.pi = pi

        self.decoder = BaseETMDecoder(
            n_latent,
            n_input)

        self.alpha_encoder = BaseETMEncoder(
            n_input,
            n_latent
        )

    @auto_move_data
    def inference(self, x):
        x_ = torch.log1p(x)

        qz_m = self.mean_encoder(x_)
        qz_v = self.var_encoder(x_)

        # depending on support of alpha, need to change transformation
        qz_a = self.alpha_encoder(x_)

        z = Normal(
            qz_a * qz_m,
            qz_a * (1 - qz_a) * qz_m ** 2 + qz_a * qz_v)

        return dict(qz_m = qz_m, qz_v = qz_v, qz_a = qz_a, z = z)

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
        qz_a = inference_outputs["qz_a"]

        # This might not work, Multinomial has default `total_count=1` which seems to cause problems
        # But according to the documentation, this is supposed to work
        log_lik = Multinomial(total_count = int(torch.max(library)), logits = rho).log_prob(x)

        # based on equation, may need to tweak
        kl_divergence = (-(qz_a / 2) * (1 + torch.log(qz_v) - (torch.square(qz_m) + qz_v)) + qz_a * torch.log(self.pi / qz_a) + (1 - qz_a) * torch.log((1 - self.pi)/ (1 - qz_a))).sum(dim = 1)

        elbo = log_lik - kl_divergence
        loss = torch.mean(-elbo)

        return LossRecorder(loss, -log_lik, kl_divergence, 0.0)
