"""
This file and `data_sim/simdata_util.py` need to be moved to the main directory to run properly because of relative imports
"""

import os
import anndata as ad
import pandas as pd
import torch
from simdata_util import sim_data

seed = 123
N = 5000
G = 10000
S = 50
D_tree = 4

dir = os.path.join("data", f"simdata_seed{seed}_N{N}_G{G}_S{S}_treeD{D_tree}")

if not os.path.exists(dir):

    torch.manual_seed(seed)
    X, A, anchor_gene_mat, theta, pi, beta, node_effect, rho, rho_raw, sequence_depth, sample_name, gene_name, node_name, topic_name = sim_data(N, G, S, D_tree)

    os.mkdir(dir)

    adata = ad.AnnData(X.numpy())

    adata.obs = pd.DataFrame(sample_name, columns = ["sample_id"])
    adata.var = pd.DataFrame(gene_name, columns = ["gene"])
    adata.write_h5ad(os.path.join(dir, "sim_tree.h5ad"))

    pd.DataFrame(X.numpy(), index = sample_name, columns = gene_name).to_csv(os.path.join(dir, "counts.csv.gz"))

    pd.DataFrame(A.to_dense().numpy(), index = topic_name, columns = node_name).to_csv(os.path.join(dir, "pbt_adj.csv"))

    pd.DataFrame(anchor_gene_mat.to_dense().numpy(), index = node_name, columns = gene_name).to_csv(os.path.join(dir, "anchor_gene_mat.csv"))

    pd.DataFrame(theta.numpy(), index = sample_name, columns = topic_name).to_csv(os.path.join(dir, "theta.csv"))

    pd.DataFrame(beta.to_dense().numpy(), index = topic_name, columns = gene_name).to_csv(os.path.join(dir, "topic_beta.csv"))

    pd.DataFrame(pi.to_dense().numpy(), index = node_name, columns = gene_name).to_csv(os.path.join(dir, "node_beta.csv"))

    # pd.DataFrame(rho.numpy(), index = sample_name, columns = gene_name).to_csv(os.path.join(dir, "rho.csv.gz"))
