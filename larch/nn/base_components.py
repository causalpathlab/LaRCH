# -*- coding: utf-8 -*-
"""base components for VAE, ETM, pathway-guided connection"""
import collections
from typing import Iterable
import torch
from torch import nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import larch.util.pbt_util as tree_util


torch.backends.cudnn.benchmark = True

def identity(x):
    return x

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


class FCLayers(nn.Module):

    """
    A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_cat_list: Iterable[int] = None,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            use_activation: bool = True,
            bias: bool = True,
            inject_covariates: bool = True,
            activation_fn: nn.Module = nn.ReLU,):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor

        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``

        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x

class MaskedLinear(nn.Linear):
    """ 
    same as Linear except has a configurable mask on the weights
    """

    def __init__(
            self,
            in_features,
            out_features,
            mask,
            bias=True,):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, input):
        if self.bias is None:
            return F.linear(input, self.weight * self.mask)
        else:
            return F.linear(input, self.weight * self.mask, self.bias)

    def __init__(
            self,
            n_in: int,
            n_out: int,
            mask: torch.Tensor = None,
            mask_first: bool = True,
            n_cat_list: Iterable[int] = None,
            n_layer: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            use_activation: bool = True,
            bias: bool = True,
            inject_covariates: bool = True,
            activation_fn: nn.Module = nn.ReLU,):
        super().__init__(
            n_in=n_in,
            n_out=n_out,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=use_activation,
            bias=bias,
            inject_covariates=inject_covariates,
            activation_fn=activation_fn
        )

        self.mask = mask ## out_features, in_features

        # if mask is None:
        #     print("No mask input, use all fully connected layers")

        if mask is not None:
            if mask_first:
                layers_dim = ([n_in]
                    + [mask.shape[0]]
                    + (n_layers - 1) * [n_hidden]
                    + [n_out])
            else:
                layers_dim = ([n_in]
                    + (n_layers - 1) * [n_hidden]
                    + [mask.shape[0]]
                    + [n_out])
        else:
            layers_dim = ([n_in]
                + (n_layers - 1) * [n_hidden]
                + [n_out])

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)

        # concatinate one hot encoding to mask if available
        if cat_dim > 0:
            mask_input = torch.cat(
                (self.mask, torch.ones(cat_dim, self.mask.shape[1])),
                dim=0)
        else:
            mask_input = self.mask

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) 
                            if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

        if mask is not None:
            if mask_first:
                # change the first layer to be MaskedLinear
                self.fc_layers[0] = nn.Sequential(
                    MaskedLinear(
                        layers_dim[0] + cat_dim * self.inject_into_layer(0),
                        layers_dim[1],
                        mask_input,
                        bias=bias,
                    ),
                    # non-default params come from defaults in original Tensorflow implementation
                    nn.BatchNorm1d(layers_dim[1], momentum=0.01, eps=0.001)
                    if use_batch_norm else None,
                    nn.LayerNorm(layers_dim[1], elementwise_affine=False)
                    if use_layer_norm else None,
                    activation_fn() if use_activation else None,
                    nn.Dropout(p=dropout_rate)
                    if dropout_rate > 0 else None,
                )
            else:
                # change the last layer to be Masked Linear
                self.fc_layers[-1] = nn.Sequential(
                    MaskedLinear(
                        layers_dim[-2] + cat_dim * self.inject_into_layer(0),
                        layers_dim[-1],
                        torch.transpose(mask_input, 0, 1),
                        bias=bias

                    ),
                    # non-default params come from defaults in original TensorFlow implementation
                    nn.BatchNorm1d(layers_dim[-1], momentum=0.01, eps=0.001)
                    if use_batch_norm else None, 
                    nn.LayerNorm(layers_dim[-1], elementwise_affine=False)
                    if use_layer_norm else None,
                    activation_fn() if use_activation else None,
                    nn.Dropout(p=dropout_rate) 
                    if dropout_rate > 0 else None,
                )

    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = [] # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided don't match init. params."
            )

        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError(
                    "cat not provided while n_cat != 0 in init. params."
                )
            if n_cat > 1: # n_cat = 1 ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x],
                                dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if ((isinstance(layer, nn.Linear) 
                                or isinstance(layer, MaskedLinear))
                            and self.inject_into_layer(i)):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list

                            x = torch.cat(
                                (x, *one_hot_cat_list_layer), 
                                dim=-1
                            )
                        x = layer(x)

        return x

class BayesianETMEncoder(nn.Module):
    """
    BayesianETM Encoder
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_hidden: int = 128,
            n_layers_individual: int = 3,
            use_batch_norm: bool = True,
            log_variational: bool = True,):
        super().__init__()
        self.log_variational = log_variational

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=None,
            n_layers=n_layers_individual,
            n_hidden=n_hidden,
            dropout_rate=0,
            use_batch_norm = use_batch_norm
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        # TODO: layer normalizaton
        if self.log_variational:
            x_ = torch.log(1 + x)

        q = self.encoder(x_, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -4.0, 4.0)/2.)
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, latent

class BayesianETMDecoder(nn.Module):
    """
    Decoder for Bayesian ETM model
    b ~ n(0, tau)
    q(b) = N(mu, nu)
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            v0=1):
        super().__init__()

        # dimensions
        self.n_input = n_input # topics
        self.n_output = n_output # genes

        # hyperparameters
        self.lnvar_0 = nn.Parameter(torch.log(torch.ones(1) * v0), requires_grad=False)

        # model parameters
        self.bias_d = nn.Parameter(torch.zeros(1, n_output))
        self.slab_mean = nn.Parameter(
            torch.randn(n_input, n_output) * torch.sqrt(torch.ones(1) * v0)
            )
        self.slab_lnvar = nn.Parameter(
            torch.ones(n_input, n_output) * torch.log(torch.ones(1) * v0)
            )

        # helper functions
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
            self,
            z: torch.Tensor):
        theta = self.soft_max(z)
        beta = self.get_beta(
            self.slab_mean,
            self.slab_lnvar,
            self.bias_d)
        beta_kl = self.sparse_kl_loss(
            self.lnvar_0,
            self.slab_mean,
            self.slab_lnvar)

        return beta, beta_kl, theta

    def get_beta(
            self,
            slab_mean: torch.Tensor,
            slab_lnvar: torch.Tensor,
            bias_d: torch.Tensor,):
        # normalize betas per node, keep track of node means and 
        mean = slab_mean

        var = torch.exp(slab_lnvar)

        eps = torch.randn_like(var)

        return mean + eps * torch.sqrt(var) - bias_d

    def safe_exp(self, x, x_min=-10, x_max=10):
        return torch.exp(torch.clamp(x, x_min, x_max))

    def soft_max(
            self,
            z: torch.Tensor):
        return torch.exp(self.log_softmax(z))

    def sparse_kl_loss(
            self,
            lnvar_0,
            slab_mean: torch.Tensor,
            slab_lnvar: torch.Tensor):
        ## Gaussian KL btwn N(μ,ν) and N(0, v0)    
        ## D_kl(q(b) || p(b)) = -1/2 * [1 + ln(var/var_0) - 1/var_0 * (mean^2 + var)]
        sq_term = (torch.exp(-lnvar_0) 
            * (torch.square(slab_mean) + torch.exp(slab_lnvar)))
        kl = -0.5 * (1. + slab_lnvar - lnvar_0 - sq_term)

        return torch.sum(kl)

class SpikeSlabDecoder(BayesianETMDecoder):
    """
    Decoder for spike and slab
    b ~ pi * N(0, tau) + (1 - pi) * d_0(b)
    q(b) = alpha * N(mu, nu)
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            pip0=0.1,
            v0=1):
        super().__init__(
            n_input=n_input,
            n_output=n_output,
            v0=v0
        )

        self.logit_0 = nn.Parameter(
            torch.logit(torch.ones(1) * pip0, eps=1e-6), requires_grad=False
            )

        self.spike_logit = nn.Parameter(
            torch.zeros(n_input, n_output) * self.logit_0
            )

    def forward(
            self, 
            z: torch.Tensor):
        theta = self.safe_exp(z) # relax to just positive, exp(z)? clamped?
        beta = self.get_beta(
            self.spike_logit, 
            self.slab_mean, 
            self.slab_lnvar, 
            self.bias_d
        ) # exp? clamped
        aa = torch.mm(theta, beta) # no exp

        beta_kl = self.sparse_kl_loss(
            self.logit_0, self.lnvar_0, 
            self.spike_logit, self.slab_mean, self.slab_lnvar)

        return beta, beta_kl, theta, aa

    def get_pip(self, spike_logit: torch.Tensor):
        return torch.sigmoid(spike_logit)

    def get_beta(
            self,
            spike_logit: torch.Tensor,
            slab_mean: torch.Tensor,
            slab_lnvar: torch.Tensor,
            bias_d: torch.Tensor):
        # standardize slab mean first
        pip = self.get_pip(spike_logit)

        mean = slab_mean * pip
        var = pip * (1 - pip) * torch.square(slab_mean)
        var = var + pip * torch.exp(slab_lnvar)

        eps = torch.randn_like(var)
        return self.safe_exp(mean + eps * torch.sqrt(var) - bias_d)

    def sparse_kl_loss(
            self,
            logit_0,
            lnvar_0,
            spike_logit: torch.Tensor,
            slab_mean: torch.Tensor,
            slab_lnvar: torch.Tensor,):
        ## PIP KL between p and p0
        ## p * ln(p / p0) + (1-p) * ln(1-p/1-p0)
        ## = p * ln(p / 1-p) + ln(1-p) +
        ##   p * ln(1-p0 / p0) - ln(1-p0)
        ## = sigmoid(logit) * logit - softplus(logit)
        ##   - sigmoid(logit) * logit0 + softplus(logit0)
        pip_hat = self.get_pip(spike_logit)
        kl_pip = pip_hat * (spike_logit - logit_0)
        kl_pip = (kl_pip 
            - nn.functional.softplus(spike_logit) 
            + nn.functional.softplus(logit_0))

        ## Gaussian KL btwn N(μ,ν) and N(0, v0)
        sq_term = (torch.exp(-lnvar_0) 
            * (torch.square(slab_mean) + torch.exp(slab_lnvar)))
        kl_g = -0.5 * (1. + slab_lnvar - lnvar_0 - sq_term)

        ## Combine both logit and Gaussian KL
        return torch.sum(kl_pip + pip_hat * kl_g)

class TreeDecoder(SpikeSlabDecoder):
    """
    Decoder for tree spike and slab ETM
    b_tg = sum_j A_tj b'_jg
    b ~ pi * N(0, tau) + (1 - pi) * d_0(b)
    q(b) = alpha * N(mu, nu)
    """

    def __init__(
            self,
            n_output: int,
            pip0=0.1,
            v0=1,
            tree_depth=3,):
        ## dimensions
        self.tree_depth = tree_depth
        self.num_tree_leaves = tree_util.pbt_depth_to_leaves(self.tree_depth)
        self.num_tree_nodes = tree_util.num_pbt_nodes(self.num_tree_leaves)

        super().__init__(
            n_output=n_output,
            n_input=self.num_tree_nodes,
            pip0=pip0,
            v0=v0
        )

        # adjacency matrix for binary tree
        self.A = nn.Parameter(
            tree_util.pbt_adj(self.tree_depth).to_dense(), requires_grad=False
            )

    def forward(
            self,
            z: torch.Tensor,):
        theta = self.safe_exp(z)
        beta = self.get_beta(
            self.spike_logit, 
            self.slab_mean, 
            self.slab_lnvar, 
            self.bias_d
        )
        aa = torch.mm(theta, torch.mm(self.A, beta))

        beta_kl = self.sparse_kl_loss(
            self.logit_0, self.lnvar_0, 
            self.spike_logit, self.slab_mean, self.slab_lnvar
        )

        return beta, beta_kl, theta, aa

class StickTreeDecoder(TreeDecoder):
    """
    Decoder for Tree ETM with added stick breaking restriction on pip
    """

    def __init__(
            self,
            n_output: int,
            pip0=0.1,
            v0=1,
            tree_depth=3,):
        super().__init__(
            n_output=n_output,
            pip0=pip0,
            v0=v0,
            tree_depth=tree_depth
        )

    def get_pip(
            self,
            spike_logit: torch.Tensor):
        sftpls_logit = nn.functional.softplus(spike_logit)

        return (torch.sigmoid(spike_logit) 
            * self.safe_exp(
                -torch.cumsum(sftpls_logit, dim=0) + sftpls_logit,
                x_min=-5, x_max=5))

class SoftmaxSpikeSlabTreeDecoder(TreeDecoder):
    """
    Decoder for Spike Slab Tree model,
    with a softmax regularizer on the spike components
    """

    def __init__(
            self,
            n_output: int,
            pip0=0.1,
            v0=1,
            tree_depth=3,):
        super().__init__(
            n_output=n_output,
            pip0=pip0,
            v0=v0,
            tree_depth=tree_depth
            )

        # Helper function: Column wise log softmax to get softmax across topics per gene
        self.cw_log_softmax = nn.LogSoftmax(dim=0)

    def get_pip(self, spike_logit: torch.Tensor):
        return torch.exp(self.cw_log_softmax(spike_logit))

    def sparse_kl_loss(
            self,
            logit_0,
            lnvar_0,
            spike_logit: torch.Tensor,
            slab_mean: torch.Tensor,
            slab_lnvar: torch.Tensor,):
        ## PIP KL between α and α0
        ## - α * ln(α) 
        ## = - soft_max(logit) * log_softmax(logit)
        pip_hat  = self.get_pip(spike_logit)
        kl_pip = - pip_hat * self.cw_log_softmax(spike_logit)

        ## Gaussian KL between  N(μ, ν) and N(0, v0)
        sq_term = (torch.exp(-lnvar_0)
            * (torch.square(slab_mean) + torch.exp(slab_lnvar)))
        kl_g = -0.5 * (1. + slab_lnvar - lnvar_0 - sq_term)

        ## Combine both logit and Gaussian KL
        return torch.sum(kl_pip + pip_hat * kl_g) # return a number sum over [N_topics, N_genes]

class SuSiEDecoder(TreeDecoder):
    """
    Decoder for the Sum of Single Effects Tree ETM
    """

    def __init__(
            self,
            n_output: int,
            v0=1,
            tree_depth=3,):
        super().__init__(
            n_output=n_output,
            v0=v0,
            tree_depth=tree_depth
        )

        # Unnormalized logit to select relevant genes for each node
        self.spike_logit = nn.Parameter(
            torch.randn(self.num_tree_nodes, n_output)
        )

    def get_pip(self, untran_pi: torch.Tensor):
        return self.soft_max(untran_pi)

    def sparse_kl_loss(
            self,
            logit_0,
            lnvar_0,
            untran_pi,
            slab_mean,
            slab_lnvar):
        # entropy term
        pi = self.get_pip(untran_pi)
        entropy = -pi * torch.log(pi)

        ## Gaussian KL between N(μ,ν) and N(0, v0)
        sq_term = (torch.exp(-lnvar_0)
            * (torch.square(slab_mean) + torch.exp(slab_lnvar)))
        kl_g = -0.5 * (1. + slab_lnvar - lnvar_0 - sq_term)

        ## Combine both entropy and Gaussian KL
        return torch.sum(entropy + pi * kl_g) # return a number sum over [N_nodes, N_genes]
