import os
from simdata_util import sim_real

N = 5000
noise = 50
seed = 123
bulk_file = "data/mean_tpm_merged.csv"
outfile = os.path.join("data", f"sim_real_N{N}_noise{noise}_seed{seed}.h5ad")

sim_real(N, bulk_file, outfile, noise = noise, seed = seed)
