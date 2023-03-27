import os
from scipy.sparse import csr_matrix
from larch.util.util import setup_anndata
import scanpy as sc
import pandas as pd
import argparse
from pytorch_lightning.loggers import CSVLogger
import datetime
from pytorch_lightning import seed_everything
from larch.run.modelhub import tree_spike_slab

parser = argparse.ArgumentParser(description='Parameters for NN')
parser.add_argument('--tree_depth', type=int, help='tree depth', default=5) # 4, 32, 128
parser.add_argument('--pip0', type=float, help='pip0', default=0.1) # 1e-3, 1e-2, 1e-1, 1
parser.add_argument('--EPOCHS', type=int, help='EPOCHS', default=2000) # 1000
parser.add_argument('--lr', type=float, help='learning_rate', default=1e-2) # 0.01
parser.add_argument('--bs', type=int, help='Batch size', default=128) # 128
parser.add_argument('--kl_weight', type=float,
                    help='weight for kl local term', default=1) # 1
parser.add_argument('--kl_weight_beta', type=float,
                    help='weight for global parameter beta in the kl term', default=1) # 1
parser.add_argument('--train_size', type=float,
                    help='set to 1 to use full dataset for training; set to 0.9 for train(0.9)/test(0.1) split',
                    default=1)
parser.add_argument('--seed', type=int, help='seed', default=66)
parser.add_argument('--use_gpu', type=int, help='which GPU to use', default=0)
parser.add_argument('--check_val_every_n_epoch', type=int,
                    help='interval to perform evalutions', default=1)
parser.add_argument('--data_file', help='filepath to h5ad file', default='data/sim_tree.h5ad')
parser.add_argument('--data_id', help='data id', default='sim_data')
args = parser.parse_args()
print(args)

model_id = f"tree_spike_slab_{args.data_id}_ep{args.EPOCHS}_treeD{args.tree_depth}_bs{args.bs}_lr{args.lr}_train_size{args.train_size}_pip{args.pip0}_kl{args.kl_weight}_klbeta{args.kl_weight_beta}_seed{args.seed}"
print(model_id)
#%%
adata = sc.read(args.data_file)
adata.layers["counts"] = csr_matrix(adata.X).copy()
setup_anndata(adata, layer="counts")

#%% Initialize the model and train
now = datetime.datetime.now()
logger = CSVLogger(save_dir = "logs", name=model_id, version = now.strftime('%Y%m%d'))
model_kwargs = {"lr": args.lr, 'use_gpu':args.use_gpu, 'train_size':args.train_size}

model = tree_spike_slab(adata, tree_depth = args.tree_depth, pip0_rho=args.pip0, kl_weight_beta = args.kl_weight_beta, kl_weight = args.kl_weight)

seed_everything(args.seed, workers=True)
#set deterministic=True for reproducibility
if os.path.exists(os.path.join("models", model_id)):
    print("Model already exists, skip training")
    print(f"Model saved at:", os.path.join("models", model_id))
else:
    print("Model does not exist, training new model")
    model.train(
        args.EPOCHS,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        batch_size=args.bs,
        logger = logger,
        deterministic=True,
        **model_kwargs,
        )
    model.save(os.path.join("models", model_id), overwrite=True, save_anndata=False)
    print(f"Model saved at:", os.path.join("models", model_id))
#%% save output
# spike, slab, standard deviation
    print("---Saving global parameters: spike, slab, standard deviation---\n")
    model.get_parameters(save_dir = os.path.join("models", model_id), overwrite = False)
    topics_np = model.get_latent_representation(deterministic=True, output_softmax_z=True)
    # topic proportions (after softmax)
    print("---Saving topic proportions (after softmax)---\n")
    topics_df = pd.DataFrame(topics_np, index= model.adata.obs.index, columns = ['topic_' + str(j) for j in range(topics_np.shape[1])])
    topics_df.to_csv(os.path.join("models", model_id,"topics.csv"))
