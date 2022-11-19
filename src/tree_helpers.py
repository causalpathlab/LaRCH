import torch

def num_pbt_nodes(kk):
    tree_depth = torch.ceil(torch.log2(kk)) + 1
    return (2**tree_depth - 1)

def pbt_nodes_to_depth(N):
    return torch.log2(N + 1)

def pbt_depth_to_leaves(D):
    return (2**(D-1))

def pbt_depth_to_nodes(D):
    return (2**D - 1)

def pbt_adj(D, signed=False):
    if (D < 2): return(torch.tensor([[1]]))

    N = pbt_depth_to_nodes(D)
    nodes = torch.arange(N)
    n_leaves = pbt_depth_to_leaves(D)
    leaves = torch.arange(n_leaves)
    bot = torch.arange(n_leaves-1, N)
    print(bot)

    idx = torch.zeros(2, n_leaves * D)
    idx[0,] = leaves.repeat(D)
    elem = torch.ones(n_leaves * D)

    idx[1,:n_leaves] = bot

    for d in range(1, D):
        bot = torch.floor((bot-1)/2)
        print(bot)
        print(d*n_leaves, (d+1)*n_leaves)
        if signed:
            elem[d*n_leaves:(d+1)*n_leaves] = 2 * (torch.ceil((leaves+1)/2**(d-1)) % 2) - 1
        idx[1, d*n_leaves:(d+1)*n_leaves] = bot

    return torch.sparse_coo_tensor(idx, elem)

print(pbt_adj(3, True))
