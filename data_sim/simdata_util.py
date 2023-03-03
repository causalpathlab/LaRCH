import os
import torch
from torch.distributions import Normal, Dirichlet, Multinomial, NegativeBinomial, Uniform
from nn import pbt_util as tree_util

softmax = torch.nn.Softmax(dim = -1)

def sim_data(N, G, S, D_tree, gamma0=500, alpha0=5):
    # Construct adjacency matrix for tree with depth D_tree

    A = tree_util.pbt_adj(D_tree)
    topics = A.size(dim = 0)
    nodes = A.size(dim = 1)

    sample_name = ["s" + str(sample) for sample in range(1, N + 1)]
    gene_name = ["g" + str(gene) for gene in range(1, G + 1)]
    node_name = ["node" + str(node) for node in range(nodes)]
    topic_name = ["topic" + str(topic) for topic in range(topics)]

    X = torch.zeros((N, G))
    pi_anchor_vals = torch.zeros(S * nodes)
    anchor_gene_vals = torch.ones_like(pi_anchor_vals)

    anchor_gene_idx_row = torch.zeros_like(pi_anchor_vals)
    anchor_gene_idx_col = torch.zeros_like(anchor_gene_idx_row)

    for node in range(nodes):
        idx_start = node * S
        idx_end = (node + 1) * S
        pi_anchor_vals[idx_start:idx_end] = Dirichlet(gamma0 / S * torch.ones(S)).sample()
        anchor_gene_idx_col[idx_start:idx_end] = torch.randperm(G)[:S]
        anchor_gene_idx_row[idx_start:idx_end] = node

    pi = torch.sparse_coo_tensor(
        indices = torch.stack([anchor_gene_idx_row, anchor_gene_idx_col],0),
        values = pi_anchor_vals,
        size = (nodes, G))
    anchor_gene_mat = torch.sparse_coo_tensor(
        indices = torch.stack([anchor_gene_idx_row, anchor_gene_idx_col],0),
        values = anchor_gene_vals,
        size = (nodes, G))

    node_effect = torch.zeros(nodes)

    for layer in range(D_tree):
        node_effect[2**layer - 1 : (2**(layer + 1) - 1)] = torch.empty(2**layer).normal_(mean = 0, std = (layer + 1) * 0.5) + torch.randn(2**layer)

    # Aggregate tree-node-specific topics to construct beta
    beta = torch.sparse.mm(A, (node_effect[:,None] * pi.to_dense()).to_sparse())
    # Sample topic proportions, this should be altered to something not as flat.
    # theta = Dirichlet(alpha0 * softmax(alpha0 * torch.randn((N, topics)))).sample()

    theta = Dirichlet(alpha0 * softmax(torch.randn((topics)))).rsample((N,))

    rho_raw = torch.mm(theta.to_sparse(), beta)

    rho = softmax(rho_raw.to_dense())

    D = torch.exp(torch.randn((N,)))

    for i in range(N):
        X[i,:] = Multinomial(round(D[i].item() * G), rho[i,:]).sample()

    return (X, A, anchor_gene_mat, theta, pi, beta, node_effect, rho, rho_raw, D, sample_name, gene_name, node_name, topic_name)

def sim_data_NB(N, G, S, D_tree, alpha0=5):
    A = tree_util.pbt_adj(D_tree)
    topics = A.size(dim = 0)
    nodes = A.size(dim = 1)

    sample_name = ["s" + str(sample) for sample in range(1, N + 1)]
    gene_name = ["g" + str(gene) for gene in range(1, G + 1)]
    node_name = ["node" + str(node) for node in range(nodes)]
    topic_name = ["topic" + str(topic) for topic in range(topics)]

    anchor_gene_vals = torch.ones(S * nodes)

    anchor_gene_idx_row = torch.zeros_like(anchor_gene_vals)
    anchor_gene_idx_col = torch.zeros_like(anchor_gene_idx_row)

    for node in range(nodes):
        idx_start = node * S
        idx_end = (node + 1) * S
        anchor_gene_idx_col[idx_start:idx_end] = torch.randperm(G)[:S]
        anchor_gene_idx_row[idx_start:idx_end] = node

    pi_anchor_vals = Uniform(torch.tensor([0.0]), torch.tensor([1.0])).rsample((S * nodes,)).reshape((S * nodes,))

    pi = torch.sparse_coo_tensor(
        indices = torch.stack([anchor_gene_idx_row, anchor_gene_idx_col],0),
        values = pi_anchor_vals,
        size = (nodes, G))
    anchor_gene_mat = torch.sparse_coo_tensor(
        indices = torch.stack([anchor_gene_idx_row, anchor_gene_idx_col],0),
        values = anchor_gene_vals,
        size = (nodes, G))

    node_effect = torch.zeros(nodes)

    for layer in range(D_tree):
        node_effect[2**layer - 1 : (2**(layer + 1) - 1)] = torch.empty(2**layer).normal_(mean = 0, std = (layer + 1) * 0.5) + torch.randn(2**layer)

    beta = torch.sparse.mm(A, (node_effect[:, None] * pi.to_dense()).to_sparse())

    theta = Dirichlet(alpha0 * softmax(torch.randn((topics)))).rsample((N,))

    rho = torch.mm(theta.to_sparse(), beta)

    D = torch.exp(torch.randn(N,G))

    X = NegativeBinomial(D, logits = rho.to_dense()).sample()

    return (X, A, anchor_gene_mat, theta, pi, beta, node_effect, rho, D, sample_name, gene_name, node_name, topic_name)
