import os
import argparse
from larch.util.simdata_util import sim_rho

def main():
    parser = argparse.ArgumentParser(description='Data simulation parameters')
    parser.add_argument('--N', type=int, help='Number of cells', default=5000)
    parser.add_argument('--noise', type=float, help='Noise parameter', default=0.1)
    parser.add_argument('--seed', type=int, help='seed', default=123)
    parser.add_argument('--bulk_file', help='filepath to bulk expression file', default='data/mean_tpm_merged.csv')
    parser.add_argument('--out_dir', help='output directory for simulated data', default='data')

    args = parser.parse_args()
    print(args)

    N = args.N
    rho = args.noise
    seed = args.seed
    bulk_file = args.bulk_file
    out_dir = args.out_dir

    outfile = os.path.join(out_dir, f"sim_rho_N{N}_rho{rho}_seed{seed}.h5ad")
    if os.path.exists(outfile):
        print("simulated data already exists at {outfile}")
    else:
        adata = sim_rho(N, bulk_file, outfile, rho=noise, seed=seed)

        print(f"saving simulated data to {outfile}")
        adata.write_h5ad(outfile)

if __name__ == "__main__":
    main()
