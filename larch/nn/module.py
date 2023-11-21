# -*- coding: utf-8 -*-
"""Main module."""
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from typing import Tuple
from larch.nn.base_model import BaseModuleClass, LossRecorder, auto_move_data
from larch.nn.base_components import (
    BayesianETMEncoder, 
    BayesianETMDecoder,
    SpikeSlabDecoder, 
    TreeBayesianDecoder,
    TreeSpikeSlabDecoder, 
    TreeStickSlabDecoder, 
    SuSiEDecoder, 
    SoftmaxSpikeSlabTreeDecoder, 
    StandardBetaDecoder,
    FullTreeSpikeSlabDecoder
)
from larch.util.util import _CONSTANTS

torch.backends.cudnn.benchmark = True

class BaseModule(BaseModuleClass):
    """
    Base Module without any decoder
    """
    def __init__(
            self,
            n_genes: int,
            n_latent: int = 32,
            n_layers_encoder_individual: int = 2,
            dim_hidden_encoder: int = 128,
            log_variational: bool = True,
            pip0_rho: float = 0.1, 
            v0: float = 1,
            kl_weight: float = 1.0,
            kl_weight_beta: float = 1.0,
            a0: float = 1e-4):
        super().__init__()

        self.n_input = n_genes
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.pip0_rho = pip0_rho
        self.v0 = v0
        self.kl_weight = kl_weight
        self.kl_weight_beta = kl_weight_beta
        self.a0 = a0

        self.z_encoder = BayesianETMEncoder(
            n_input=self.n_input,
            n_output=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            log_variational=self.log_variational,
        )

    def dir_llik(
            self,
            xx: torch.Tensor,
            aa: torch.Tensor,
            a0: float = 1e-4) -> torch.Tensor:
        '''
        # Dirichlet log-likelihood:
        # lgamma(sum a) - lgamma(sum a + x)
        # sum lgamma(a + x) - lgamma(a)
        # @param xx [batch_size, n_genes]
        # @param aa [batch_size, n_genes]
        # @return log-likelihood
        '''
        reconstruction_loss = None

        term1 = (torch.lgamma(torch.sum(aa + a0, dim=-1))
            - torch.lgamma(torch.sum(aa + a0 + xx, dim=-1))) # [n_batch]

        term2 = torch.sum(
            torch.where(
                xx > 0, 
                torch.lgamma(aa + a0 + xx)
                - torch.lgamma(aa + a0),
                torch.zeros_like(xx)),
            dim=-1
        ) # [n_batch]

        reconstruction_loss = term1 + term2 # [n_batch]
        return reconstruction_loss

    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        return dict(z=z)

    @auto_move_data
    def inference(self, x: torch.Tensor) -> dict:
        x_ = x

        qz_m, qz_v, z = self.z_encoder(x_)

        return dict(qz_m=qz_m, qz_v=qz_v, z=z)

    @auto_move_data
    def generative(self, z) -> dict:
        beta, beta_kl, theta, aa = self.decoder(z)

        return dict(beta=beta, beta_kl=beta_kl, theta=theta, aa=aa)

    def sample_from_posterior_z(
            self,
            x: torch.Tensor,
            deterministic: bool = True,
            output_softmax_z: bool = True,):
        inference_out = self.inference(x)
        
        if deterministic:
            z = inference_out["qz_m"]
        else: 
            z = inference_out["z"]

        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["theta"]

        return dict(z=z)

    @auto_move_data
    def get_reconstruction_loss(
            self,
            x: torch.Tensor,) -> torch.Tensor:
        """
        Returns the
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``

        Returns
        -------
        type
            tensor of means of the scaled frequencies
        """
        inference_out = self.inference(x)
        z = inference_out["z"]

        gen_out = self.generative(z)
        aa = gen_out["aa"]

        reconstruction_loss = -self.dir_llik(x, aa, self.a0)

        return reconstruction_loss

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Return the reconstruction loss and the Kullback divergences.
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        local_l_mean
            tensor of means of the prior distribution of latent variable l
            with shape (batch_size, 1)
        local_l_var
            tensor of variances of the prior distribution of latent variable l
            with shape (batch_size, 1)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape (batch_size, n_labels)
        mode
            indicates which head/tail to use in the joint network
        Returns
        -------
        the reconstruction loss and the Kullback divergences
        """
        kl_weight_beta = self.kl_weight_beta
        kl_weight = self.kl_weight

        x = tensors[_CONSTANTS.X_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        # kl_divergence for beta, beta_kl, tensor of torch.size([]) <- torch.sum([N_topics, N_genes])
        kl_divergence_beta = generative_outputs["beta_kl"]

        # [batch_size]
        reconstruction_loss = self.get_reconstruction_loss(x)

        # KL divergence for z [batch_size]
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_local = kl(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(mean, scale)).sum(dim=1) # summing over all latent dimensions
        
        loss = (torch.mean(reconstruction_loss + kl_weight * kl_local)
            + kl_weight_beta * kl_divergence_beta / x.shape[1])

        return LossRecorder(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local=kl_local,
            kl_beta=kl_divergence_beta
        )

class FlatModule(BaseModule):
    """
    Generic module for flat topic models
    """
    def __init__(
            self,
            n_genes: int,
            n_latent: int = 32,
            decoder="ssl",
            n_layers_encoder_individual: int = 2,
            dim_hidden_encoder: int = 128,
            log_variational: bool = True,
            pip0_rho: float = 0.1, 
            v0: float = 1,
            kl_weight: float = 1.0,
            kl_weight_beta: float = 1.0,
            a0: float = 1e-4,):

        super().__init__(
            n_genes=n_genes,
            n_latent=n_latent,
            n_layers_encoder_individual=n_layers_encoder_individual,
            dim_hidden_encoder=dim_hidden_encoder,
            log_variational=log_variational,
            pip0_rho=pip0_rho,
            v0=v0,
            kl_weight=kl_weight,
            kl_weight_beta=kl_weight_beta,
            a0=a0,
        )

        if decoder == "ssl":
            self.decoder = SpikeSlabDecoder(
                n_input=self.n_latent,
                n_output=self.n_input,
                pip0=self.pip0_rho,
                v0=self.v0
            )
        elif decoder == "bayesian":
            self.decoder = BayesianETMDecoder(
                n_input=self.n_latent,
                n_output=self.n_input,
                v0=self.v0
            )
        else: raise ValueError("Invalid decoder")

class TreeModule(BaseModule):
    """
    Generic module for tree structured topic models
    """

    def __init__(
            self,
            n_genes: int,
            tree_depth: int = 3,
            decoder="ssl",
            n_layers_encoder_individual: int = 2,
            dim_hidden_encoder: int = 128,
            log_variational: bool = True,
            pip0_rho: float = 0.1,
            v0: float = 1,
            kl_weight: float = 1.0,
            kl_weight_beta: float = 1.0,
            a0: float = 1e-4,):
        if decoder == "full_ssl":
            n_latent = 2 ** tree_depth - 1
        else:
            n_latent = 2 ** (tree_depth - 1)

        super().__init__(
            n_genes=n_genes,
            n_latent=n_latent,
            n_layers_encoder_individual=n_layers_encoder_individual,
            dim_hidden_encoder=dim_hidden_encoder,
            log_variational=log_variational,
            pip0_rho=pip0_rho,
            kl_weight=kl_weight,
            kl_weight_beta=kl_weight_beta,
            a0=a0,
        )
        self.tree_depth = tree_depth

        if decoder == "ssl":
            self.decoder = TreeSpikeSlabDecoder(
                n_output=self.n_input,
                pip0=self.pip0_rho,
                v0=self.v0,
                tree_depth=self.tree_depth
            )

        elif decoder == "bayesian":
            self.decoder = TreeBayesianDecoder(
                n_output=self.n_input,
                v0=self.v0,
                tree_depth=self.tree_depth
            )

        elif decoder == "full_ssl":
            self.decoder = FullTreeSpikeSlabDecoder(
                n_output=self.n_input,
                pip0=self.pip0_rho,
                v0=self.v0,
                tree_depth=self.tree_depth
            )

        elif decoder == "stick":
            self.decoder = TreeStickSlabDecoder(
                n_output=self.n_input,
                pip0=self.pip0_rho,
                v0=self.v0,
                tree_depth=self.tree_depth
            )

        elif decoder == "softmax":
            self.decoder = SoftmaxSpikeSlabTreeDecoder(
                n_output=self.n_input,
                pip0=self.pip0_rho,
                v0=self.v0,
                tree_depth=self.tree_depth
            )

        elif decoder == "susie":
            self.decoder = SuSiEDecoder(
                n_output=self.n_input,
                v0=self.v0,
                tree_depth=self.tree_depth
            )

        else: raise ValueError("Invalid decoder")

