#######################################################
## helper functions related to a perfect binary tree ##
#######################################################
import math
import torch

#' @param N total number of tree nodes
#' @return tree depth
def num_pbt_nodes(kk):
    tree_depth = math.ceil(math.log2(kk)) + 1
    return int(2**tree_depth - 1) # number of tree nodes

def pbt_nodes_to_depth(N):
    return int(math.log2(N + 1))

#' @param D tree depth
def pbt_depth_to_leaves(D):
    return 2**(D-1)

#' Perfect Binary Tree-based convolution matrix
#' @param D tree depth
#' @return S adjacency matrix (#leaves x #tree nodes)
def pbt_adj_helper(D, _signed=False):
    if D < 2: 
        elem = torch.ones(1)
        col = torch.zeros(1)
        row = torch.zeros(1)
    else:   
        _N = 2**D - 1
        _nodes = torch.arange(_N)
        _leaves = torch.arange(2**(D-1))
        _bot = torch.arange(2**(D-1)-1, _N)
        col = _bot
        row = _leaves
        elem = torch.ones(len(_leaves))

        for d in range(1, D):
            _bot = torch.floor((_bot+1)/2)-1
            if _signed:
                _new_elem = 2*torch.ceil(_leaves/2**(d-1)) % 2 - 1
            else:
                _new_elem = torch.ones(len(_leaves))

            elem = torch.cat((elem, _new_elem))
            col = torch.cat((col, _bot)).int()
            row = torch.cat((row, _leaves)).int()

    return elem, col, row

def pbt_adj(D, _signed=False):
    _elem, _col, _row = pbt_adj_helper(D, _signed)

    out = torch.sparse_coo_tensor(torch.stack([_row, _col],0), _elem)
    return out

def pbt_full_adj(D, _signed=False):
    _elem, _col, _row = pbt_adj_helper(1, _signed)
    if D > 1:
        for d in range(2, D + 1):
            _elem_d, _col_d, _row_d = pbt_adj_helper(d, _signed)

            _row_d = _row_d + (2**(d-1) - 1)

            _elem = torch.cat((_elem, _elem_d))
            _col = torch.cat((_col, _col_d)).int()
            _row = torch.cat((_row, _row_d)).int()

    out = torch.sparse_coo_tensor(torch.stack([_row, _col], 0), _elem)

    return out

# example
#A = pbt_adj(3)
#A.to_dense()
#X = torch.rand(7,3)
#X
#torch.mm(A.to_dense(), X)

# 1 if node j is parent of node i, 0 otherwise
def pbt_parent(D):
    if D < 2: return torch.zeros(1)
    _N = 2**D - 1

    _elem = torch.ones(_N - 1)
    _row = torch.arange(start = 1, end = _N)
    _col = torch.floor((_row - 1) / 2)

    out = torch.sparse_coo_tensor(torch.stack([_row, _col], 0), _elem, size=(_N, _N))
    
    out = out.to_dense()

    return out.numpy()
