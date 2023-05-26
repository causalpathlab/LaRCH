import logging
import os
from typing import List, Optional, Union
import torch
from torch import nn as nn
import numpy as np
from anndata import AnnData
from larch.util.util import DataSplitter, TrainRunner, BaseModelClass
from larch.nn.TrainingPlan import TrainingPlan
from larch.nn.module import TreeSpikeSlabModule, BALSAMModule, SuSiETreeModule, TreeStickSlabModule

logger = logging.getLogger(__name__)

def _unpack_tensors(tensors):
    x = tensors["X"].squeeze_(0)
    return x

class TreeSpikeSlab(BaseModelClass):
    """
    tree spike slab

    Parameters
    ----------
    adata_seq
        Spliced and unspliced count AnnData object that has been registered via :func:`data.setup_anndata`
        and contains data.
    tree_depth
        depth of the tree
    **model_kwargs
        Keyword args for :class:`~module.TreeSpikeSlabModule`

    Examples
    --------
    """

    def __init__(
        self,
        adata_seq: AnnData,
        tree_depth: int = 3,
        **model_kwargs,
    ):
        super().__init__()
        self.adata = adata_seq
        self.tree_depth = tree_depth
        self.module = TreeSpikeSlabModule(
            n_genes = self.adata.n_vars,
            #n_latent=n_latent,
            tree_depth = self.tree_depth,
            **model_kwargs,
        )

        self._model_summary_string = (
            "tree_spike_slab with the following params:  n_genes: {}, tree_depth: {}"
        ).format(self.adata.n_vars, self.tree_depth)

    def train(
        self,
        max_epochs: Optional[int] = 1000,
        lr: float = 1e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.

        """

        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )

        update_dict = {
            "lr": lr,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **kwargs,
        )
        return runner()

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData = None,
        deterministic: bool = True,
        output_softmax_z: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset,

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        output_softmax_z
            if true, output probability, otherwise output z.
        batch_size
            Minibatch size for data loading into model.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()

        latent_z = []
        for tensors in scdl:
            (
                sample_batch
            ) = _unpack_tensors(tensors)
            z_dict  = self.module.sample_from_posterior_z(sample_batch, deterministic=deterministic, output_softmax_z=output_softmax_z)
            latent_z.append(z_dict["z"])

        latent_z = torch.cat(latent_z).cpu().detach().numpy()

        print(f'Deterministic: {deterministic}, output_softmax_z: {output_softmax_z}' )
        return latent_z

    @torch.no_grad()
    def get_parameters(
        self,
        save_dir = None,
        overwrite = False,
    ) -> List[np.ndarray]:
        """return the spike logit, slab mean, slab lnvar for rho"""

        self.module.eval()
        decoder = self.module.decoder


        if not os.path.exists(os.path.join(save_dir,"model_parameters")) or overwrite:
            os.makedirs(os.path.join(save_dir,"model_parameters"), exist_ok=overwrite)


        np.savetxt(os.path.join(
                save_dir,"model_parameters", "spike_logit_rho.txt"
            ), decoder.spike_logit.cpu().numpy())


        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_mean_rho.txt"
            ), decoder.slab_mean.cpu().numpy())


        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_lnvar_rho.txt"
            ), decoder.slab_lnvar.cpu().numpy())


    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            save_path = os.path.join(
                dir_path, "adata.h5ad"
            )
            self.adata.write(save_path)

        varnames_save_path = os.path.join(
            dir_path, "var_names.csv"
        )
        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")

        torch.save(self.module.state_dict(), model_save_path)

    def load_state_dict(self, state):
        self.module.load_state_dict(state)

class SpikeSlab(BaseModelClass):
    """
    """

    def __init__(
        self,
        adata_seq: AnnData,
        n_latent: int = 32,
        **model_kwargs,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.adata = adata_seq

        self.module = BALSAMModule(
            n_genes = self.adata.n_vars,
            n_latent=n_latent,
            **model_kwargs,
        )

        self._model_summary_string = (
            "spike_slab with the following params: \nn_latent: {},  n_genes: {}"
        ).format(n_latent, self.adata.n_vars)

    def train(
        self,
        max_epochs: Optional[int] = 1000,
        lr: float = 1e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        """

        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )

        update_dict = {
            "lr": lr,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **kwargs,
        )
        return runner()

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData = None,
        deterministic: bool = True,
        output_softmax_z: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset,
        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        output_softmax_z
            if true, output probability, otherwise output z.
        batch_size
            Minibatch size for data loading into model.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()

        latent_z = []
        for tensors in scdl:
            (
                sample_batch
            ) = _unpack_tensors(tensors)
            z_dict  = self.module.sample_from_posterior_z(sample_batch, deterministic=deterministic, output_softmax_z=output_softmax_z)
            latent_z.append(z_dict["z"])

        latent_z = torch.cat(latent_z).cpu().detach().numpy()

        print(f'Deterministic: {deterministic}, output_softmax_z: {output_softmax_z}' )
        return latent_z


    @torch.no_grad()
    def get_parameters(
        self,
        save_dir = None,
        overwrite = False,
    ) -> List[np.ndarray]:
        """return the spike logit, slab mean, slab lnvar for rho"""

        self.module.eval()
        decoder = self.module.decoder


        if not os.path.exists(os.path.join(save_dir,"model_parameters")) or overwrite:
            os.makedirs(os.path.join(save_dir,"model_parameters"), exist_ok=overwrite)


        np.savetxt(os.path.join(
                save_dir,"model_parameters", "spike_logit_rho.txt"
            ), decoder.spike_logit.cpu().numpy())


        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_mean_rho.txt"
            ), decoder.slab_mean.cpu().numpy())


        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_lnvar_rho.txt"
            ), decoder.slab_lnvar.cpu().numpy())


        np.savetxt(os.path.join(
                save_dir,"model_parameters", "bias_gene.txt"
            ), decoder.bias_d.cpu().numpy())

    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.
        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.
        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            save_path = os.path.join(
                dir_path, "adata.h5ad"
            )
            self.adata.write(save_path)
        varnames_save_path = os.path.join(
            dir_path, "var_names.csv"
        )

        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")

        torch.save(self.module.state_dict(), model_save_path)

class SuSiETree(BaseModelClass):
    """
    susie_tree model for single-cell data.

    Parameters
    ----------
    adata_seq
        Spliced and unspliced count AnnData object that has been registered via :func:`data.setup_anndata`
        and contains data.
    tree_depth
        depth of the tree
    **model_kwargs
        Keyword args for :class:`~module.DeltaETM_module`

    Examples
    --------
    """

    def __init__(
        self,
        adata_seq: AnnData,
        tree_depth: int = 3,
        **model_kwargs,
    ):
        super().__init__()
        self.adata = adata_seq
        self.tree_depth = tree_depth
        self.module = SuSiETreeModule(
            n_genes = self.adata.n_vars,
            #n_latent=n_latent,
            tree_depth = self.tree_depth,
            **model_kwargs,
        )

        self._model_summary_string = (
            "susie_tree with the following params:  n_genes: {}, tree_depth: {}"
        ).format(self.adata.n_vars, self.tree_depth)

    def train(
        self,
        max_epochs: Optional[int] = 1000,
        lr: float = 1e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.

        """

        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )

        update_dict = {
            "lr": lr,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **kwargs,
        )
        return runner()

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData = None,
        deterministic: bool = True,
        output_softmax_z: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset,

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        output_softmax_z
            if true, output probability, otherwise output z.
        batch_size
            Minibatch size for data loading into model.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()

        latent_z = []
        for tensors in scdl:
            (
                sample_batch
            ) = _unpack_tensors(tensors)
            z_dict  = self.module.sample_from_posterior_z(sample_batch, deterministic=deterministic, output_softmax_z=output_softmax_z)
            latent_z.append(z_dict["z"])

        latent_z = torch.cat(latent_z).cpu().detach().numpy()

        print(f'Deterministic: {deterministic}, output_softmax_z: {output_softmax_z}' )
        return latent_z

    @torch.no_grad()
    def get_parameters(
        self,
        save_dir = None,
        overwrite = False,
    ) -> List[np.ndarray]:
        """return the spike logit, slab mean, slab lnvar for rho"""

        self.module.eval()
        decoder = self.module.decoder


        if not os.path.exists(os.path.join(save_dir,"model_parameters")) or overwrite:
            os.makedirs(os.path.join(save_dir,"model_parameters"), exist_ok=overwrite)


        np.savetxt(os.path.join(
                save_dir,"model_parameters", "untran_pi.txt"
            ), decoder.untran_pi.cpu().numpy())



        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_mean_rho.txt"
            ), decoder.slab_mean.cpu().numpy())


        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_lnvar_rho.txt"
            ), decoder.slab_lnvar.cpu().numpy())


    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            save_path = os.path.join(
                dir_path, "adata.h5ad"
            )
            self.adata.write(save_path)

        varnames_save_path = os.path.join(
            dir_path, "var_names.csv"
        )
        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")

        torch.save(self.module.state_dict(), model_save_path)

class TreeStickSlab(BaseModelClass):
    """
    tree stick slab

    Parameters
    ----------
    adata_seq
        RNA count AnnData object registered via :func:`data.setup_anndata`
        and contains data
    tree_depth
        depth of the tree
    **model_kwargs
        Keyword args for :class:`~module.TreeStickSlabModule`
    """

    def __init__(
        self,
        adata_seq: AnnData,
        tree_depth: int = 3,
        **model_kwargs,
    ):
        super().__init__()

        self.adata = adata_seq
        self.tree_depth = tree_depth
        self.module = TreeStickSlabModule(
            n_genes = self.adata.n_vars,
            tree_depth = self.tree_depth,
            **model_kwargs)

        self._model_summary_string = (
            "tree_stick_slab with the following params: n_genes: {}, tree_depth: {}"
        ).format(self.adata.n_vars, self.tree_depth)

    def train(
        self,
        max_epochs: Optional[int] = 1000,
        lr: float = 1e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.

        """

        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )

        update_dict = {
            "lr": lr,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **kwargs,
        )
        return runner()

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData = None,
        deterministic: bool = True,
        output_softmax_z: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset,

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        output_softmax_z
            if true, output probability, otherwise output z.
        batch_size
            Minibatch size for data loading into model.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()

        latent_z = []
        for tensors in scdl:
            (
                sample_batch
            ) = _unpack_tensors(tensors)
            z_dict  = self.module.sample_from_posterior_z(sample_batch, deterministic=deterministic, output_softmax_z=output_softmax_z)
            latent_z.append(z_dict["z"])

        latent_z = torch.cat(latent_z).cpu().detach().numpy()

        print(f'Deterministic: {deterministic}, output_softmax_z: {output_softmax_z}' )
        return latent_z

    @torch.no_grad()
    def get_parameters(
        self,
        save_dir = None,
        overwrite = False,
    ) -> List[np.ndarray]:
        """ return spike logit, slab mean, slab lnvar for rho"""

        self.module.eval()
        decoder = self.module.decoder

        if not os.path.exists(os.path.join(save_dir, "model_parameters")) or overwrite:
            os.makedirs(os.path.join(save_dir, "model_parameters"), exist_ok=overwrite)

        alpha_logit = decoder.spike_logit.cpu()
        logit = torch.logit(
            torch.sigmoid(alpha_logit) *
            torch.exp(torch.clamp(-torch.cumsum(nn.functional.softplus(alpha_logit), dim = 0)) *
            (1 + torch.exp(alpha_logit)), min = -5, max = 5)
        )

        np.savetxt(os.path.join(
            save_dir, "model_parameters", "spike_logit_rho.txt"
        ), logit.numpy())


        np.savetxt(os.path.join(
            save_dir,"model_parameters", "slab_mean_rho.txt"
        ), decoder.slab_mean.cpu().numpy())


        np.savetxt(os.path.join(
            save_dir,"model_parameters", "slab_lnvar_rho.txt"
        ), decoder.slab_lnvar.cpu().numpy())

    def save(
        self,
        dir_path: str,
        overwrite: bool=False,
        save_anndata: bool=False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            save_path = os.path.join(
                dir_path, "adata.h5ad"
            )
            self.adata.write(save_path)

        varnames_save_path = os.path.join(
            dir_path, "var_names.csv"
        )
        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")

        torch.save(self.module.state_dict(), model_save_path)

    def load_state_dict(self, state):
        self.module.load_state_dict(state)
