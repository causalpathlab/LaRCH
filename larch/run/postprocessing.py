import numpy as np
import os
import math
import argparse
from scipy.special import logit, expit
from scipy.stats import norm
from larch.util import pbt_util

def main():
    parser = argparse.ArgumentParser(description="Postprocessing parameters")

    parser.add_argument("--cutoff", type=int, help="Cutoff metric, number of significant genes required to keep a node", default = 200)
    parser.add_argument("--dir", help="Directory containing data files", default = "/Users/pattieye/Documents/YPP/SuSiEVAE/models/sim_real_noise50")
    parser.add_argument('--use_unique', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    print(args)
    cutoff = args.cutoff

    dir = args.dir
    mean_rho = np.genfromtxt(os.path.join(dir, "slab_mean_rho.txt"))
    lnvar_rho = np.genfromtxt(os.path.join(dir, "slab_lnvar_rho.txt"))
    logit_pi = np.genfromtxt(os.path.join(dir, "spike_logit_rho.txt"))

    nodes = len(mean_rho)
    D = pbt_util.pbt_nodes_to_depth(nodes)

    def ssl_pval(mean_rho, lnvar_rho, logit_pi):
        pip = expit(logit_pi)
        var = pip * (1 - pip) * np.square(mean_rho)
        var = var + pip * np.exp(lnvar_rho)

        norm_p = 2 * norm.sf(np.abs(pip * mean_rho), scale = np.sqrt(var))

        return(pip * norm_p + (1 - pip))

    pvals = ssl_pval(mean_rho, lnvar_rho, logit_pi)
    sig_bool = pvals < 0.05

    parent_mtx = pbt_util.pbt_parent(D)

    for node in range(nodes - 1, 0, -1):
        parent = math.floor((node - 1) / 2)
        if args.use_unique:
            n_sig = np.sum(sig_bool[node,:] ^ (sig_bool[parent,:] & sig_bool[node,:]))
        else:
            n_sig = np.sum(sig_bool[node,:])

        if n_sig < cutoff:
            # remove node from tree
            if np.sum(parent_mtx[:, node]) > 0:
                # children of node become children of node's parent
                parent_mtx[:, parent] = parent_mtx[:, parent] + parent_mtx[:, node]

                parent_mtx[:, node] = np.zeros(nodes)

            parent_mtx[node, parent] = 0

        elif np.sum(parent_mtx[:, parent]) == 1:
            # node is the sole child of parent, combine into one
            _mean_rho = mean_rho[parent, :] + mean_rho[node, :]

            _lnvar_rho = np.log(np.exp(lnvar_rho[parent, :]) + np.exp(lnvar_rho[node, :]))

            _logit_pi = np.maximum(logit_pi[parent, :], logit_pi[node, :])

            _pvals = ssl_pval(_mean_rho, _lnvar_rho, _logit_pi)
            _sig_bool = _pvals < 0.05

            mean_rho[parent, :] = _mean_rho
            lnvar_rho[parent, :] = _lnvar_rho
            logit_pi[parent, :] = _logit_pi
            pvals[parent, :] = _pvals
            sig_bool[parent, :] = _sig_bool

            sig_bool[parent, :] = sig_bool[parent, :] | sig_bool[node, :]

            if np.sum(parent_mtx[:, node]) > 0:
                # children of node become children of node's parent
                parent_mtx[:, parent] = parent_mtx[:, node]
                parent_mtx[:, node] = np.zeros(nodes)

            parent_mtx[node, parent] = 0

    np.savetxt(os.path.join(dir, "tree_mtx.txt"), parent_mtx, delimiter=",")
    np.savetxt(os.path.join(dir, "postprocessed_slab_mean_rho.txt"), mean_rho, delimiter=",")
    np.savetxt(os.path.join(dir, "postprocessed_slab_lnvar_rho.txt"), lnvar_rho, delimiter=",")
    np.savetxt(os.path.join(dir, "postprocessed_spike_logit_rho.txt"), logit_pi, delimiter=",")

if __name__ == "__main__":
    main()
