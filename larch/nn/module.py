# -*- coding: utf-8 -*-
"""Main module."""
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from typing import Tuple
from larch.nn.base_model import BaseModuleClass, LossRecorder, auto_move_data
from larch.nn.base_components import BayesianETMEncoder, TreeDecoder, StickTreeDecoder, BALSAMDecoder, BALSAMEncoder, SusieDecoder
from larch.util.util import _CONSTANTS

torch.backends.cudnn.benchmark = True

class tree_spike_slab_module(BaseModuleClass):
    """
    Tree VAE
    """

    def __init__(
        self,
        n_genes: int,
        #n_latent: int = 32,
        tree_depth: int = 3,
        n_layers_encoder_individual: int = 2,
        dim_hidden_encoder: int = 128,
        log_variational: bool = True,
        pip0_rho: float = 0.1,
        kl_weight: float = 1.0,
        kl_weight_beta: float = 1.0,
    ):
        super().__init__()

        self.n_input = n_genes
        self.n_latent = 2**(tree_depth-1)
        self.log_variational = log_variational
        self.pip0_rho = pip0_rho
        self.kl_weight = kl_weight
        self.kl_weight_beta = kl_weight_beta
        self.tree_depth = tree_depth

        self.z_encoder = BayesianETMEncoder(
            n_input=self.n_input,
            n_output=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            log_variational = self.log_variational,
        )

        self.decoder = TreeDecoder(self.n_input, # n_genes
                                   pip0 = self.pip0_rho,
                                   tree_depth=self.tree_depth
                                )

    def dir_llik(self,
                 xx: torch.Tensor,
                 aa: torch.Tensor,
    ) -> torch.Tensor:
        '''
        # Dirichlet log-likelihood:
        # lgamma(sum a) - lgamma(sum a + x)
        # sum lgamma(a + x) - lgamma(a)
        # @param xx [batch_size, n_genes]
        # @param aa [batch_size, n_genes]
        # @return log-likelihood
        '''
        reconstruction_loss = None

        term1 = (torch.lgamma(torch.sum(aa, dim=-1)) -
                torch.lgamma(torch.sum(aa + xx, dim=-1))) #[n_batch]
        term2 = torch.sum(torch.where(xx > 0,
                            torch.lgamma(aa + xx) -
                            torch.lgamma(aa),
                            torch.zeros_like(xx)),
                            dim=-1) #[n_batch
        reconstruction_loss = term1 + term2 #[n_batch
        return reconstruction_loss

    def _get_inference_input(self, tensors):
        return dict(x=tensors["X"])

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

        beta, rho_kl, theta, rho, aa  = self.decoder(z)
        return dict(rho = rho, rho_kl = rho_kl, theta = theta, beta = beta, aa = aa)

    def sample_from_posterior_z(
        self,
        x: torch.Tensor,
        deterministic: bool = True,
        output_softmax_z: bool = True,
    ):
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
        x: torch.Tensor,
    ) -> torch.Tensor:
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

        reconstruction_loss = -self.dir_llik(x, aa)

        return reconstruction_loss

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs, # this is important to include
        kl_weight=1.0,
        kl_weight_beta = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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
        x = tensors["X"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        rho_kl = generative_outputs["rho_kl"]

        # [batch_size]
        reconstruction_loss = self.get_reconstruction_loss(x)

        # KL Divergence for z [batch_size]
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        ) # suming over all the latent dimensinos
        # kl_divergence for beta, rho_kl, tensor of torch.size([]) <- torch.sum([N_topics, N_genes])
        kl_divergence_beta = rho_kl
        kl_local = kl_divergence_z

        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) + kl_weight_beta * kl_divergence_beta/x.shape[1]

        return LossRecorder(loss, reconstruction_loss, kl_local,
                            reconstruction_loss_spliced=reconstruction_loss,
                            reconstruction_loss_unspliced=torch.Tensor(0),
                            kl_beta = kl_divergence_beta,
                            kl_rho = rho_kl,
                            kl_delta = torch.Tensor(0))

class BALSAM_module(BaseModuleClass):
    """
    The BETM module
    """

    def __init__(
        self,
        n_genes: int,
        n_latent: int = 32,
        n_layers_encoder_individual: int = 2,
        dim_hidden_encoder: int = 128,
        log_variational: bool = True,
        pip0_rho: float = 0.1,
        kl_weight: float = 1.0,
        kl_weight_beta: float = 1.0,
    ):
        super().__init__()

        self.n_input = n_genes
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.pip0_rho = pip0_rho
        self.kl_weight_beta = kl_weight_beta

        self.z_encoder = BALSAMEncoder(
            n_input=self.n_input,
            n_output=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            log_variational = self.log_variational,
        )

        self.decoder = BALSAMDecoder(self.n_latent ,
                                    self.n_input,
                                    pip0 = self.pip0_rho,
                                    )

    def dir_llik(self,
                 xx: torch.Tensor,
                 aa: torch.Tensor,
    ) -> torch.Tensor:
        '''
        # Dirichlet log-likelihood:
        # lgamma(sum a) - lgamma(sum a + x)
        # sum lgamma(a + x) - lgamma(a)
        # @param xx [batch_size, n_genes]
        # @param aa [batch_size, n_genes]
        # @return log-likelihood
        '''
        reconstruction_loss = None

        term1 = (torch.lgamma(torch.sum(aa, dim=-1)) -
                torch.lgamma(torch.sum(aa + xx, dim=-1))) #[n_batch]
        term2 = torch.sum(torch.where(xx > 0,
                            torch.lgamma(aa + xx) -
                            torch.lgamma(aa),
                            torch.zeros_like(xx)),
                            dim=-1) #[n_batch
        reconstruction_loss = term1 + term2 #[n_batch
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

        rho, rho_kl, theta  = self.decoder(z)

        return dict(rho = rho, rho_kl = rho_kl, theta = theta)

    def sample_from_posterior_z(
        self,
        x: torch.Tensor,
        deterministic: bool = True,
        output_softmax_z: bool = True,
    ):
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
        x: torch.Tensor,
    ) -> torch.Tensor:
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
        theta = gen_out["theta"]

        rho = gen_out["rho"]
        log_aa = torch.clamp(torch.mm(theta, rho), -10, 10)
        aa = torch.exp(log_aa)

        reconstruction_loss = -self.dir_llik(x, aa)

        return reconstruction_loss

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs, # this is important to include
        kl_weight=1.0,
        #kl_weight_beta = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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
        x = tensors[_CONSTANTS.X_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        rho_kl = generative_outputs["rho_kl"]

        # [batch_size]
        reconstruction_loss = self.get_reconstruction_loss(x)

        # KL Divergence for z [batch_size]
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        ) # suming over all the latent dimensinos
        # kl_divergence for beta, rho_kl, tensor of torch.size([]) <- torch.sum([N_topics, N_genes])
        kl_divergence_beta = rho_kl
        kl_local = kl_divergence_z

        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) + kl_weight_beta * kl_divergence_beta/x.shape[1]

        return LossRecorder(loss, reconstruction_loss, kl_local,
                            reconstruction_loss_spliced=reconstruction_loss,
                            reconstruction_loss_unspliced=torch.Tensor(0),
                            kl_beta = kl_divergence_beta,
                            kl_rho = rho_kl,
                            kl_delta = torch.Tensor(0))

class susie_tree_module(BaseModuleClass):
    """
    Tree VAE
    """

    def __init__(
        self,
        n_genes: int,
        #n_latent: int = 32,
        tree_depth: int = 3,
        n_layers_encoder_individual: int = 2,
        dim_hidden_encoder: int = 128,
        log_variational: bool = True,
        pip0_rho: float = 0.1,
        kl_weight: float = 1.0,
        kl_weight_beta: float = 1.0,
    ):
        super().__init__()

        self.n_input = n_genes
        self.n_latent = 2**(tree_depth-1)
        self.log_variational = log_variational
        self.pip0_rho = pip0_rho
        self.kl_weight = kl_weight
        self.kl_weight_beta = kl_weight_beta
        self.tree_depth = tree_depth

        self.z_encoder = BayesianETMEncoder(
            n_input=self.n_input,
            n_output=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            log_variational = self.log_variational,
        )
        self.decoder = SusieDecoder(self.n_input,
                                   tree_depth=self.tree_depth
                                )

    def dir_llik(self,
                 xx: torch.Tensor,
                 aa: torch.Tensor,
    ) -> torch.Tensor:
        '''
        # Dirichlet log-likelihood:
        # lgamma(sum a) - lgamma(sum a + x)
        # sum lgamma(a + x) - lgamma(a)
        # @param xx [batch_size, n_genes]
        # @param aa [batch_size, n_genes]
        # @return log-likelihood
        '''
        reconstruction_loss = None

        term1 = (torch.lgamma(torch.sum(aa, dim=-1)) -
                torch.lgamma(torch.sum(aa + xx, dim=-1))) #[n_batch]
        term2 = torch.sum(torch.where(xx > 0,
                            torch.lgamma(aa + xx) -
                            torch.lgamma(aa),
                            torch.zeros_like(xx)),
                            dim=-1) #[n_batch
        reconstruction_loss = term1 + term2 #[n_batch
        return reconstruction_loss

    def _get_inference_input(self, tensors):
        return dict(x=tensors["X"])

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

        rho, rho_kl, theta, beta, aa  = self.decoder(z)
        return dict(rho = rho, rho_kl = rho_kl, theta = theta, beta = beta, aa = aa)

    def sample_from_posterior_z(
        self,
        x: torch.Tensor,
        deterministic: bool = True,
        output_softmax_z: bool = True,
    ):
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
        x: torch.Tensor,
    ) -> torch.Tensor:
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

        reconstruction_loss = -self.dir_llik(x, aa)

        return reconstruction_loss

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs, # this is important to include
        kl_weight=1.0,
        kl_weight_beta = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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
        x = tensors["X"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        rho_kl = generative_outputs["rho_kl"]

        # [batch_size]
        reconstruction_loss = self.get_reconstruction_loss(x)

        # KL Divergence for z [batch_size]
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        ) # suming over all the latent dimensinos
        # kl_divergence for beta, rho_kl, tensor of torch.size([]) <- torch.sum([N_topics, N_genes])
        kl_divergence_beta = rho_kl
        kl_local = kl_divergence_z

        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) + kl_weight_beta * kl_divergence_beta/x.shape[1]

        return LossRecorder(loss, reconstruction_loss, kl_local,
                            reconstruction_loss_spliced=reconstruction_loss,
                            reconstruction_loss_unspliced=torch.Tensor(0),
                            kl_beta = kl_divergence_beta,
                            kl_rho = rho_kl,
                            kl_delta = torch.Tensor(0))

class tree_stick_slab_module(tree_spike_slab_module):
    """
    Tree VAE with stick breaking pip
    """

    def __init__(
        self,
        n_genes: int,
        tree_depth: int = 3,
        n_layers_encoder_individual: int = 2,
        dim_hidden_encoder: int = 128,
        log_variational: bool = True,
        alpha0_rho: float = 0.1,
        kl_weight: float = 1.0,
        kl_weight_beta: float = 1.0,
    ):
        super().__init__(
            n_genes: int,
            tree_depth: int = 3,
            n_layers_encoder_individual: int = 2,
            dim_hidden_encoder: int = 128,
            log_variational: bool = True,
            alpha0_rho: float = 0.1,
            kl_weight: float = 1.0,
            kl_weight_beta: float = 1.0,
        )

        self.decoder = StickTreeDecoder(
            self.n_input,
            alpha0 = self.alpha0_rho,
            tree_depth = self.tree_depth
        )
