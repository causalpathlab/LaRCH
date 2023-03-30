#######################################################
## helper functions related to a perfect binary tree ##
#######################################################
import math
import torch

#' @param N total number of tree nodes
#' @return tree depth
def num_pbt_nodes(kk):
    tree_depth = math.ceil(math.log2(kk)) + 1
    return (2**tree_depth - 1).astype(int) # number of tree nodes

def pbt_nodes_to_depth(N):
    return (math.log2(N + 1)).astype(int)

#' @param D tree depth
def pbt_depth_to_leaves(D):
    return 2**(D-1)

#' Perfect Binary Tree-based convolution matrix
#' @param D tree depth
#' @return S adjacency matrix (#leaves x #tree nodes)
def pbt_adj(D, _signed = False):
    if D < 2: out = torch.ones(1)
    _N = 2**D - 1
    _nodes = torch.arange(_N)
    _leaves = torch.arange(2**(D-1))
    _bot = torch.arange(2**(D-1)-1, _N)
    _col = _bot
    _row = _leaves
    _elem = torch.ones(len(_leaves))

    for d in range(1, D):
        _bot = torch.floor((_bot+1)/2)-1
        if _signed:
            _new_elem = 2*torch.ceil(_leaves/2**(d-1)) % 2 - 1
        else:
            _new_elem = torch.ones(len(_leaves))

        _elem = torch.cat((_elem, _new_elem))
        _col = torch.cat((_col, _bot)).int()
        _row = torch.cat((_row, _leaves)).int()

    out = torch.sparse_coo_tensor(torch.stack([_row, _col],0), _elem)
    return out

# example
#A = pbt_adj(3)
#A.to_dense()
#X = torch.rand(7,3)
#X
#torch.mm(A.to_dense(), X)